"""Profile plots."""

from pathlib import Path

import streamlit as st

from qgsw.plots.heatmaps import (
    AnimatedHeatmaps,
)
from qgsw.run_summary import RunOutput

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

vars_profile = [var for var in run.output_vars if var.scope.point_wise]

var = st.selectbox("Variable to display", vars_profile)

with st.form(key="var-form"):
    level = st.selectbox("Level", list(range(levels_nb)))
    submit = st.form_submit_button("Display")

if submit:
    plot = AnimatedHeatmaps.from_point_wise_output(
        [run.folder],
        field=var.name,
        ensembles=0,
        levels=level,
    )

    fig = plot.retrieve_figure()
    fig.update_layout(height=1000, width=800)

    st.plotly_chart(plot.retrieve_figure(), use_container_width=False)
