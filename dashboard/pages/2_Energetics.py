"""Profile plots."""

from pathlib import Path

import streamlit as st

from qgsw.plots.scatter import ScatterPlot
from qgsw.run_summary import RunOutput

ROOT = Path(__file__).parent.parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.title("Profiles")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)

st.write(run)

vars_profile = [var for var in run.output_vars if var.scope.level_wise_at_most]

var = st.selectbox("Variable to display", vars_profile)

if var.scope.level_wise:
    with st.form(key="var-form-level"):
        levels_nb = run.summary.configuration.model.h.shape[0]
        ls = list(range(levels_nb))
        level = st.selectbox("Level", [*ls, "All"])
        submit = st.form_submit_button("Display")

    if submit:
        if level == "All":
            folders = [run.folder] * levels_nb
            levels = ls
        else:
            folders = [run.folder]
            levels = [level]
        plot = ScatterPlot.level_wise_from_folders(
            folders=folders,
            field=var.name,
            levels=levels,
        )
elif var.scope.ensemble_wise:
    with st.form(key="var-form-ensemble"):
        ensembles = 0
        submit = st.form_submit_button("Display")

    if submit:
        plot = ScatterPlot.ensemble_wise_from_folders(
            folders=[run.folder],
            field=var.name,
        )
if submit:
    fig = plot.retrieve_figure()
    fig.update_layout(height=750, width=500)

    st.plotly_chart(plot.retrieve_figure(), use_container_width=True)
