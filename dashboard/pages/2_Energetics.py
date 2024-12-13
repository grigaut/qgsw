"""Profile plots."""

from itertools import product
from pathlib import Path

import streamlit as st

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

st.write(run)

if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")

ens_wise = st.toggle("Display ensemble-wise variables.")

if ens_wise:
    vars_profile = [
        v for v in run.output_vars if v.scope.stricly_ensemble_wise
    ]
else:
    vars_profile = [v for v in run.output_vars if v.scope.stricly_level_wise]

selected_vars = st.multiselect("Variable to display", vars_profile)

check_unit_compatibility(*selected_vars)

if ens_wise:
    with st.form(key="var-form-ensemble-wise"):
        ensembles = [0] * len(selected_vars)
        submit = st.form_submit_button("Display")
    if submit:
        plot = ScatterPlot.ensemble_wise_from_folders(
            folders=[run.folder] * len(selected_vars),
            ensembles=ensembles,
            fields=[v.name for v in selected_vars],
        )
else:
    with st.form(key="var-form-level-wise"):
        levels_nb = run.summary.configuration.model.h.shape[0]
        levels = list(range(levels_nb))
        selected_levels = st.multiselect("Level(s)", levels)
        submit = st.form_submit_button("Display")
    if submit:
        prod = list(product(selected_vars, selected_levels))
        fs = [e[0].name for e in prod]
        ls = [e[1] for e in prod]
        plot = ScatterPlot.level_wise_from_folders(
            folders=[run.folder] * len(fs),
            fields=fs,
            ensembles=[0] * len(fs),
            levels=ls,
        )
if submit:
    fig = plot.retrieve_figure()
    fig.update_layout(height=750, width=500)

    st.plotly_chart(plot.retrieve_figure(), use_container_width=True)
