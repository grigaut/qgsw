"""Assimilation plot."""

from pathlib import Path

import streamlit as st
import torch

from qgsw.fields.scope import Scope
from qgsw.fields.variables.utils import check_unit_compatibility
from qgsw.fields.variables.variable_sets import create_qg_variable_set
from qgsw.output import RunOutput
from qgsw.plots.heatmaps import AnimatedHeatmaps
from qgsw.plots.scatter import ScatterPlot
from qgsw.specs import DEVICE

ROOT = Path(__file__).parent.parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.title("Assimilation")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)

if run.summary.configuration.simulation.kind != "assimilation":
    st.error("The selected simulation is not ran assimilation one.", icon="⚠️")
    st.stop()

config = run.summary.configuration

vars_dict = create_qg_variable_set(
    config.physics,
    config.space,
    config.model,
    torch.float64,
    DEVICE.get(),
)
st.write(run)

prefix_ref = run.summary.configuration.simulation.reference.prefix
run_ref = RunOutput(folder, prefix=prefix_ref)

vars_dict_ref = create_qg_variable_set(
    config.physics,
    config.space,
    config.simulation.reference,
    torch.float64,
    DEVICE.get(),
)
levels_nb = run.summary.configuration.model.h.shape[0]


if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")


st.title("Point-wise variables")

vars_pts = [var for var in vars_dict.values() if var.scope == Scope.POINT_WISE]

selected_var_pts = st.selectbox("Variable to display", vars_pts)
selected_var_pts_ref = vars_dict_ref[selected_var_pts.name]

with st.form(key="var-form"):
    level = st.selectbox("Level", list(range(levels_nb)))
    submit_pts = st.form_submit_button("Display")

if submit_pts:
    uvhs_ref = (output.read() for output in run_ref.outputs())
    values_ref = (selected_var_pts_ref.compute(uvh) for uvh in uvhs_ref)
    data_ref = [d[0, level].T.cpu() for d in values_ref]
    uvhs = (output.read() for output in run.outputs())
    values = (selected_var_pts.compute(uvh) for uvh in uvhs)
    data = [d[0, level].T.cpu() for d in values]

    plot_pts = AnimatedHeatmaps(
        [data_ref, data],
    )
    plot_pts.set_subplot_titles(
        [f"Reference - Level {level}", f"Model - Level {level}"],
    )
    plot_pts.set_colorbar_text(
        f"{selected_var_pts.description} [{selected_var_pts.unit.value}]",
    )
    plot_pts.set_slider_prefix("Time: ")
    plot_pts.set_figure_size(800, 1000)
    st.plotly_chart(plot_pts.retrieve_figure(), use_container_width=True)

st.title("Level-wise variables")

vars_lvl = [v for v in vars_dict.values() if v.scope == Scope.LEVEL_WISE]

selected_vars_lvl = st.multiselect("Variable to display", vars_lvl)
selected_vars_lvl_ref = [vars_dict_ref[v.name] for v in selected_vars_lvl]

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
    names = [
        f"{var.description} - Ens: 0 - Level: {level}"
        for level in selected_lvl
        for var in selected_vars_lvl
    ]
    datas_ref = [
        [
            d[0, level].cpu()
            for d in (
                var.compute(uvh)
                for uvh in (output.read() for output in run_ref.outputs())
            )
        ]
        for level in selected_lvl
        for var in selected_vars_lvl_ref
    ]
    names_ref = [
        f"{var.description} - Ens: 0 - Level: {level} (reference)"
        for level in selected_lvl
        for var in selected_vars_lvl
    ]

    plot_lvl = ScatterPlot(datas=datas_ref + datas)
    plot_lvl.set_xaxis_title("Time [s]")
    plot_lvl.set_yaxis_title(f"[{selected_vars_lvl[0].unit.value}]")
    plot_lvl.set_traces_name(*names_ref, *names)
    plot_lvl.set_figure_size(height=750, width=500)
    xs = [list(run.seconds()) for _ in levels for _ in selected_vars_lvl]
    xs_ref = [
        list(run_ref.seconds()) for _ in levels for _ in selected_vars_lvl
    ]
    plot_lvl.set_xs(*xs, *xs_ref)
    st.plotly_chart(plot_lvl.retrieve_figure(), use_container_width=True)


st.title("Ensemble-wise variables")

vars_ens = [v for v in vars_dict.values() if v.scope == Scope.ENSEMBLE_WISE]

selected_vars_ens = st.multiselect("Variable to display", vars_ens)
selected_vars_ens_ref = [vars_dict_ref[v.name] for v in selected_vars_ens]

if not check_unit_compatibility(*selected_vars_ens):
    st.error("Selected variables don't have the same unit.")
    st.stop()

with st.form(key="var-form-ensemble-wise"):
    submit_ensemble = st.form_submit_button("Display")

if submit_ensemble:
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
    names = [f"{var.description} - Ens: 0" for var in selected_vars_ens]
    datas_ref = [
        [
            d[0].cpu()
            for d in (
                var.compute(uvh)
                for uvh in (output.read() for output in run_ref.outputs())
            )
        ]
        for var in selected_vars_ens_ref
    ]
    names_ref = [
        f"{var.description} - Ens: 0 (reference)" for var in selected_vars_ens
    ]

    plot_ens = ScatterPlot(datas=datas_ref + datas)
    plot_ens.set_xaxis_title("Time [s]")
    plot_ens.set_yaxis_title(
        f"[{selected_vars_ens[0].unit.value}]",
    )
    plot_ens.set_traces_name(*names_ref, *names)
    plot_ens.set_figure_size(height=750, width=500)
    xs = [list(run.seconds()) for _ in selected_vars_ens]
    xs_ref = [list(run_ref.seconds()) for _ in selected_vars_ens]
    plot_ens.set_xs(*xs, *xs_ref)
    st.plotly_chart(plot_ens.retrieve_figure(), use_container_width=True)
