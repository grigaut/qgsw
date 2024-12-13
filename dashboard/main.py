"""Main Page."""

from pathlib import Path

import streamlit as st

from qgsw.run_summary import RunOutput

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.set_page_config("Dashboard", layout="wide")
st.title("Dashboard")

st.write("Display output from QGSW simulations.")

st.write("# Explore outputs")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)

st.write(run)

if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")
