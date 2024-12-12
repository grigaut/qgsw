"""Main Page."""

from pathlib import Path

import streamlit as st

from qgsw.plots.heatmaps import (
    AnimatedHeatmaps,
)
from qgsw.run_summary import RunOutput

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT.joinpath("output")

st.set_page_config("Dashboard", layout="centered")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)
layers_nb = run.summary.configuration.model.h.shape[0]

st.write(run)

with st.form(key="var-form"):
    var = st.selectbox("Variable to display", run.output_vars)
    layer = st.selectbox("Layer", list(range(layers_nb)))
    submit = st.form_submit_button("Display")

if submit:
    plot = AnimatedHeatmaps.from_run_folders(
        [run.folder],
        field=var.name,
        layers=layer,
    )

    fig = plot.retrieve_figure()
    fig.update_layout(height=750, width=500)

    st.plotly_chart(plot.retrieve_figure(), use_container_width=True)
