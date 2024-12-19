"""Profile plots."""

from pathlib import Path

import streamlit as st
import torch

from qgsw.output import RunOutput, add_qg_variables
from qgsw.plots.heatmaps import (
    AnimatedHeatmaps,
)
from qgsw.plots.scatter import ScatterPlot
from qgsw.specs import DEVICE
from qgsw.variables.utils import check_unit_compatibility

ROOT = Path(__file__).parent.parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.title("Profiles")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)
config = run.summary.configuration
add_qg_variables(
    run,
    config.physics,
    config.space,
    config.model,
    torch.float64,
    DEVICE.get(),
)
levels_nb = run.summary.configuration.model.h.shape[0]

st.write(run)

if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")

st.title("Point-wise variables")

vars_pts = [var for var in run.vars if var.scope.point_wise]

selected_var_pts = st.selectbox("Variable to display", vars_pts)

with st.form(key="var-form"):
    level = st.selectbox("Level", list(range(levels_nb)))
    submit_pts = st.form_submit_button("Display")

if submit_pts:
    data = [d[0, level].T.cpu() for d in run[selected_var_pts.name]]

    plot_pts = AnimatedHeatmaps(
        [data],
    )
    plot_pts.set_slider_prefix("Time: ")
    txt = f"{selected_var_pts.description}[{selected_var_pts.unit.value}]"
    plot_pts.set_colorbar_text(txt)
    plot_pts.set_subplot_titles(
        [f"{run.summary.configuration.io.name} - Level {level}"],
    )
    plot_pts.set_frame_labels([f"{t.days} days" for t in run.timesteps()])

    plot_pts.set_figure_size(height=1000, width=800)

    st.plotly_chart(plot_pts.retrieve_figure(), use_container_width=False)

st.title("Level-wise variables")

vars_lvl = [v for v in run.vars if v.scope.stricly_level_wise]

selected_vars_lvl = st.multiselect("Variable to display", vars_lvl)

if not check_unit_compatibility(*selected_vars_lvl):
    st.error("Selected variables don't have the same unit.")
    st.stop()

with st.form(key="var-form-level-wise"):
    levels_nb = run.summary.configuration.model.h.shape[0]
    levels = list(range(levels_nb))
    selected_lvl = st.multiselect("Level(s)", levels)
    submit_lvl = st.form_submit_button("Display")
if submit_lvl:
    datas = [
        [d[0, level].cpu() for d in run[var.name]]
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

vars_ens = [v for v in run.vars if v.scope.stricly_ensemble_wise]

selected_vars_ens = st.multiselect("Variable to display", vars_ens)

if not check_unit_compatibility(*selected_vars_ens):
    st.error("Selected variables don't have the same unit.")
    st.stop()

with st.form(key="var-form-ensemble-wise"):
    submit_ens = st.form_submit_button("Display")
if submit_ens:
    datas = [[d[0].cpu() for d in run[var.name]] for var in selected_vars_ens]
    plot_ens = ScatterPlot(datas)
    plot_ens.set_xaxis_title("Time [s]")
    plot_ens.set_yaxis_title(f"[{selected_vars_ens[0].unit.value}]")
    names = [f"{var.description} - Ens: 0" for var in selected_vars_ens]
    plot_ens.set_traces_name(*names)
    xs = [list(run.seconds()) for _ in selected_vars_ens]
    plot_ens.set_xs(*xs)
    plot_ens.set_figure_size(height=750, width=500)

    st.plotly_chart(plot_ens.retrieve_figure(), use_container_width=True)
