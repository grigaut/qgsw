"""Profile plots."""

from itertools import product
from pathlib import Path

import streamlit as st

from qgsw.plots.heatmaps import (
    AnimatedHeatmaps,
)
from qgsw.plots.scatter import ScatterPlot
from qgsw.run_summary import RunOutput
from qgsw.variables.utils import check_unit_compatibility

ROOT = Path(__file__).parent.parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.title("Profiles")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)
levels_nb = run.summary.configuration.model.h.shape[0]

st.write(run)

if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")

st.title("Point-wise variables")

vars_points = [var for var in run.output_vars if var.scope.point_wise]

selected_var_points = st.selectbox("Variable to display", vars_points)

with st.form(key="var-form"):
    level = st.selectbox("Level", list(range(levels_nb)))
    submit_points = st.form_submit_button("Display")

if submit_points:
    plot_points = AnimatedHeatmaps.from_point_wise_output(
        [run.folder],
        field=selected_var_points.name,
        ensembles=0,
        levels=level,
    )

    plot_points.set_figure_size(height=1000, width=800)

    st.plotly_chart(plot_points.retrieve_figure(), use_container_width=False)

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
    prod = list(product(selected_vars_levels, selected_levels))
    fs = [e[0].name for e in prod]
    ls = [e[1] for e in prod]
    plot_levels = ScatterPlot.level_wise_from_folders(
        folders=[run.folder] * len(fs),
        fields=fs,
        ensembles=[0] * len(fs),
        levels=ls,
    )
    fig_levels = plot_levels.retrieve_figure()
    fig_levels.update_layout(height=750, width=500)

    st.plotly_chart(plot_levels.retrieve_figure(), use_container_width=True)


st.title("Ensemble-wise variables")

vars_ensembles = [v for v in run.output_vars if v.scope.stricly_ensemble_wise]

selected_vars_ensembles = st.multiselect("Variable to display", vars_ensembles)

if not check_unit_compatibility(*selected_vars_ensembles):
    st.error("Selected variables don't have the same unit.")
    st.stop()

with st.form(key="var-form-ensemble-wise"):
    ensembles = [0] * len(selected_vars_ensembles)
    submit_ensemble = st.form_submit_button("Display")
if submit_ensemble:
    plot_ensembles = ScatterPlot.ensemble_wise_from_folders(
        folders=[run.folder] * len(selected_vars_ensembles),
        ensembles=ensembles,
        fields=[v.name for v in selected_vars_ensembles],
    )
    fig_ensembles = plot_ensembles.retrieve_figure()
    fig_ensembles.update_layout(height=750, width=500)

    st.plotly_chart(plot_ensembles.retrieve_figure(), use_container_width=True)
