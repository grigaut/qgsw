"""Assimilation plot."""

from pathlib import Path

import streamlit as st

from qgsw.plots.heatmaps import AnimatedHeatmaps
from qgsw.plots.scatter import ScatterPlot
from qgsw.run_summary import RunOutput
from qgsw.variables.utils import check_unit_compatibility

ROOT = Path(__file__).parent.parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.title("Assimilation")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)
prefix_ref = run.summary.configuration.simulation.reference.prefix
run_ref = RunOutput(folder, prefix=prefix_ref)
levels_nb = run.summary.configuration.model.h.shape[0]

st.write(run)

if run.summary.configuration.simulation.kind != "assimilation":
    st.error("The selected simulation is not ran assimilation one.", icon="⚠️")
    st.stop()

if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")


st.title("Point-wise variables")

vars_points = [var for var in run.output_vars if var.scope.point_wise]

selected_var_points = st.selectbox("Variable to display", vars_points)

with st.form(key="var-form"):
    level = st.selectbox("Level", list(range(levels_nb)))
    submit_points = st.form_submit_button("Display")

if submit_points:
    outputs = list(run.outputs())
    outputs_ref = list(run.outputs())
    var_name = selected_var_points.name

    plot_points = AnimatedHeatmaps(
        [
            [out.read()[var_name][0, level, ...].T for out in outputs_ref],
            [out.read()[var_name][0, level, ...].T for out in outputs],
        ],
    )
    plot_points.set_subplot_titles(
        [f"Reference - Level {level}", f"Model - Level {level}"],
    )
    plot_points.set_colorbar_text(
        f"{run[var_name].description} [{run[var_name].unit}]",
    )
    plot_points.set_slider_prefix("Time: ")
    plot_points.set_figure_size(800, 1000)
    st.plotly_chart(plot_points.retrieve_figure(), use_container_width=True)


st.title("Level-wise variables")

vars_levels = [v for v in run.output_vars if v.scope.stricly_level_wise]

selected_vars_levels = st.multiselect("Variable to display", vars_levels)

if not check_unit_compatibility(*selected_vars_levels):
    st.error("Selected variables don't have the same unit.")
    st.stop()

with st.form(key="var-form-level-wise"):
    levels_nb = run.summary.configuration.model.h.shape[0]
    levels = list(range(levels_nb))
    selected_levels = st.multiselect("Level(s)", levels)
    submit_levels = st.form_submit_button("Display")


if submit_levels:
    datas = [
        [data[0, level] for data in run[var.name].datas()]
        for level in levels
        for var in selected_vars_levels
    ]
    names = [
        f"{var.description} - Ens: 0 - Level: {level}"
        for level in levels
        for var in selected_vars_levels
    ]
    datas_ref = [
        [data[0, level] for data in run_ref[var.name].datas()]
        for level in levels
        for var in selected_vars_levels
    ]
    names_refs = [
        f"{var.description} - Ens: 0 - Level: {level} (reference)"
        for level in levels
        for var in selected_vars_levels
    ]

    plot_levels = ScatterPlot(datas=datas_ref + datas)
    plot_levels.set_xaxis_title("Time [s]")
    plot_levels.set_yaxis_title(f"[{selected_vars_levels[0].unit}]")
    plot_levels.set_traces_name(*names_refs, *names)
    plot_levels.set_figure_size(height=750, width=500)
    xs = [
        list(run[var.name].seconds())
        for _ in levels
        for var in selected_vars_levels
    ]
    xs_ref = [
        list(run_ref[var.name].seconds())
        for _ in levels
        for var in selected_vars_levels
    ]
    plot_levels.set_xs(*xs, *xs_ref)
    st.plotly_chart(plot_levels.retrieve_figure(), use_container_width=True)


st.title("Ensemble-wise variables")

vars_ensembles = [v for v in run.output_vars if v.scope.stricly_ensemble_wise]

selected_vars_ensembles = st.multiselect("Variable to display", vars_ensembles)

if not check_unit_compatibility(*selected_vars_ensembles):
    st.error("Selected variables don't have the same unit.")
    st.stop()

with st.form(key="var-form-ensemble-wise"):
    submit_ensemble = st.form_submit_button("Display")

if submit_ensemble:
    datas = [
        [data[0] for data in run[var.name].datas()]
        for var in selected_vars_ensembles
    ]
    names = [f"{var.description} - Ens: 0" for var in selected_vars_ensembles]
    datas_ref = [
        [data[0] for data in run_ref[var.name].datas()]
        for var in selected_vars_ensembles
    ]
    names_refs = [
        f"{var.description} - Ens: 0 (reference)"
        for var in selected_vars_ensembles
    ]

    plot_ensembles = ScatterPlot(datas=datas_ref + datas)
    plot_ensembles.set_xaxis_title("Time [s]")
    plot_ensembles.set_yaxis_title(f"[{selected_vars_ensembles[0].unit}]")
    plot_ensembles.set_traces_name(*names_refs, *names)
    plot_ensembles.set_figure_size(height=750, width=500)
    xs = [list(run[var.name].seconds()) for var in selected_vars_ensembles]
    xs_ref = [
        list(run_ref[var.name].seconds()) for var in selected_vars_ensembles
    ]
    plot_ensembles.set_xs(*xs, *xs_ref)

    st.plotly_chart(plot_ensembles.retrieve_figure(), use_container_width=True)
