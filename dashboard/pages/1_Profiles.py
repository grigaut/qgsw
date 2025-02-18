"""Profile plots."""

from pathlib import Path

import streamlit as st

from qgsw.fields.scope import Scope
from qgsw.fields.variables.utils import check_unit_compatibility
from qgsw.models.instantiation import get_model_class
from qgsw.models.qg.projected.modified.utils import is_modified
from qgsw.output import RunOutput
from qgsw.plots.heatmaps import (
    AnimatedHeatmaps,
)
from qgsw.plots.scatter import ScatterPlot

ROOT = Path(__file__).parent.parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.title("Profiles")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)
config = run.summary.configuration

vars_dict = get_model_class(config.model).get_variable_set(
    config.space,
    config.physics,
    config.model,
)
levels_nb = run.summary.configuration.model.h.shape[0]
if is_modified(run.summary.configuration.model.type):
    levels_nb -= 1

st.write(run)

if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")

st.title("Point-wise variables")

vars_pts = [var for var in vars_dict.values() if var.scope == Scope.POINT_WISE]

selected_var_pts = st.selectbox("Variable to display", vars_pts)

with st.form(key="var-form"):
    levels = st.multiselect("Level", list(range(levels_nb)))
    submit_pts = st.form_submit_button("Display")

if submit_pts:
    uvhs = (output.read() for output in run.outputs())
    values = [selected_var_pts.compute(uvh) for uvh in uvhs]
    datas = [[d[0, lvl].T.cpu() for d in values] for lvl in levels]

    plot_pts = AnimatedHeatmaps(
        datas,
    )
    plot_pts.set_slider_prefix("Time: ")
    txt = f"{selected_var_pts.description}[{selected_var_pts.unit.value}]"
    plot_pts.set_colorbar_text(txt)
    plot_pts.set_subplot_titles(
        [
            f"{run.summary.configuration.io.name} - Level {lvl}"
            for lvl in levels
        ],
    )
    plot_pts.set_frame_labels([f"{t.days} days" for t in run.timesteps()])

    plot_pts.set_figure_size(height=1000, width=800)

    st.plotly_chart(plot_pts.retrieve_figure(), use_container_width=False)

st.title("Level-wise variables")

vars_lvl = [v for v in vars_dict.values() if v.scope == Scope.LEVEL_WISE]

selected_vars_lvl = st.multiselect("Variable to display", vars_lvl)

if not check_unit_compatibility(*selected_vars_lvl):
    st.error("Selected variables don't have the same unit.")
    st.stop()

with st.form(key="var-form-level-wise"):
    levels = list(range(levels_nb))
    selected_lvl = st.multiselect("Level(s)", levels)
    submit_lvl = st.form_submit_button("Display")
if submit_lvl:
    datas = [
        [
            d[0, level].cpu()
            for d in (
                var.compute(uvh)
                for uvh in (output.read() for output in run.outputs())
            )
        ]
        for level in selected_lvl
        for var in selected_vars_lvl
    ]
    plot_lvl = ScatterPlot(datas)
    plot_lvl.set_xaxis_title("Time [s]")
    plot_lvl.set_yaxis_title(f"[{selected_vars_lvl[0].unit.value}]")
    names = [
        f"{var.description} - Ens: 0 - Level: {level}"
        for level in selected_lvl
        for var in selected_vars_lvl
    ]
    plot_lvl.set_traces_name(*names)
    xs = [list(run.seconds()) for _ in selected_lvl for _ in selected_vars_lvl]
    plot_lvl.set_xs(*xs)
    plot_lvl.set_figure_size(height=750, width=500)

    st.plotly_chart(plot_lvl.retrieve_figure(), use_container_width=True)

st.title("Ensemble-wise variables")

vars_ens = [v for v in vars_dict.values() if v.scope == Scope.ENSEMBLE_WISE]

selected_vars_ens = st.multiselect("Variable to display", vars_ens)

if not check_unit_compatibility(*selected_vars_ens):
    st.error("Selected variables don't have the same unit.")
    st.stop()

with st.form(key="var-form-ensemble-wise"):
    submit_ens = st.form_submit_button("Display")
if submit_ens:
    datas = [
        [
            d[0].cpu()
            for d in (
                var.compute(uvh)
                for uvh in (output.read() for output in run.outputs())
            )
        ]
        for var in selected_vars_ens
    ]
    plot_ens = ScatterPlot(datas)
    plot_ens.set_xaxis_title("Time [s]")
    plot_ens.set_yaxis_title(f"[{selected_vars_ens[0].unit.value}]")
    names = [f"{var.description} - Ens: 0" for var in selected_vars_ens]
    plot_ens.set_traces_name(*names)
    xs = [list(run.seconds()) for _ in selected_vars_ens]
    plot_ens.set_xs(*xs)
    plot_ens.set_figure_size(height=750, width=500)

    st.plotly_chart(plot_ens.retrieve_figure(), use_container_width=True)
