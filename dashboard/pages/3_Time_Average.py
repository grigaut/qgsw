"""Profile plots."""

from pathlib import Path

import streamlit as st
import torch

from qgsw.fields.errors.error_sets import create_errors_set
from qgsw.fields.scope import Scope
from qgsw.models.instantiation import get_model_class
from qgsw.models.qg.modified.utils import is_modified
from qgsw.output import RunOutput
from qgsw.plots.heatmaps import (
    AnimatedHeatmaps,
)
from qgsw.simulation.names import SimulationName

ROOT = Path(__file__).parent.parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.title("Time Average 2D Fields")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)

config = run.summary.configuration
vars_dict = get_model_class(config.model).get_variable_set(
    config.space,
    config.physics,
    config.model,
)
levels_nb = run.summary.configuration.model.h.shape[0]
if is_modified(config.model.type):
    levels_nb -= 1

st.write(run)

if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")


st.title("Point-wise variables")

if run.summary.configuration.simulation.type == SimulationName.ASSIMILATION:
    show_error_pts = st.toggle("Display Errors", key="toggle-pts")

    if show_error_pts:
        errors = create_errors_set()
        model_config_ref = run.summary.configuration.simulation.reference
        run_ref = RunOutput(folder, model_config=model_config_ref)
        model_ref_config = config.simulation.reference
        vars_dict_ref = get_model_class(model_ref_config).get_variable_set(
            config.space,
            config.physics,
            model_ref_config,
        )
        error_type_pts = st.selectbox(
            "Error to display",
            [e for e in errors.values() if e.get_scope() == Scope.POINT_WISE],
            format_func=lambda x: x.get_description(),
            key="sbox-pts",
        )


vars_pts = [var for var in vars_dict.values() if var.scope == Scope.POINT_WISE]

selected_var_pts = st.selectbox("Variable to display", vars_pts)

if (
    run.summary.configuration.simulation.type == SimulationName.ASSIMILATION
    and show_error_pts
):
    selected_var_pts_ref = vars_dict_ref[selected_var_pts.name]
    error_pts = error_type_pts(selected_var_pts, selected_var_pts_ref)
    error_pts.slices = [slice(None, None), slice(0, 1), ...]


with st.form(key="var-form"):
    levels = st.multiselect("Level", list(range(levels_nb)))
    submit_pts = st.form_submit_button("Display")

if submit_pts:
    uvhs = [(output.read() for output in run.outputs()) for _ in levels]
    if (
        run.summary.configuration.simulation.type
        == SimulationName.ASSIMILATION
        and show_error_pts
    ):
        uvhs_ref = [
            (output.read() for output in run_ref.outputs()) for _ in levels
        ]
        datas = [
            [
                torch.mean(
                    torch.stack(
                        [
                            error_pts.compute_point_wise(uvh, uvh_ref)[
                                0,
                                level,
                            ].T.cpu()
                            for uvh, uvh_ref in zip(uvhs[k], uvhs_ref[k])
                        ],
                        dim=0,
                    ),
                    dim=0,
                ),
            ]
            for k, level in enumerate(levels)
        ]
        plot_pts = AnimatedHeatmaps(
            datas,
        )
        plot_pts.set_subplot_titles(
            [f"Error - Level {level}" for level in levels],
        )
    else:
        values = [
            (selected_var_pts.compute(uvh) for uvh in uvh_gen)
            for uvh_gen in uvhs
        ]
        datas = [
            [
                torch.mean(
                    torch.stack([d[0, level].T.cpu() for d in values[k]]),
                    dim=0,
                ),
            ]
            for k, level in enumerate(levels)
        ]

        plot_pts = AnimatedHeatmaps(
            datas,
        )
        plot_pts.set_subplot_titles(
            [
                f"{run.summary.configuration.io.name} - Level {level}"
                for level in levels
            ],
        )
    plot_pts.set_slider_prefix("Time: ")
    txt = f"{selected_var_pts.description}[{selected_var_pts.unit.value}]"
    plot_pts.set_colorbar_text(txt)

    plot_pts.set_figure_size(height=1000, width=800 * min(3, len(levels)))

    st.plotly_chart(plot_pts.retrieve_figure(), use_container_width=False)
