"""Browser results files."""

import re
from pathlib import Path

import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from qgsw.scripts.output import build_output_folder
from qgsw.utils.storage import get_path_from_env

sns.set_theme("notebook", style="dark")

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT.joinpath("output")

storage = get_path_from_env("STORAGE")
g5k_storage = get_path_from_env("G5K_IMPORT_STORAGE")

st.set_page_config("Dashboard", layout="wide")

st.title("Dashboard")

outputs = list(storage.glob("*[0-9]{8}_[0-9]{6}-*")) + list(
    g5k_storage.glob("*")
)

pattern = re.compile(".*-[0-9]{8}_[0-9]{6}-[a-zA-Z0-9]{8}")

output_source = (
    st.segmented_control(
        "Data location",
        [g5k_storage, storage],
        format_func=lambda p: p.name,
        default=g5k_storage,
    )
    / "param_optim"
)

matching_folders = sorted(
    item
    for item in output_source.iterdir()
    if item.is_dir() and pattern.match(item.name)
)


folder = st.selectbox(
    "Select a folder",
    matching_folders,
    index=None,
    format_func=lambda p: p.name,
)


if folder is not None:
    st.header(f"Experiment details: {folder.name}")

    output_folder = build_output_folder(folder)

    st.subheader("Parameters")
    params = output_folder.params.load()

    model_type = params["model_type"]

    with st.expander("Parameters JSON", expanded=False):
        st.write(params)

    st.subheader("Optimization results")
    optim = output_folder.optim.load()

    with st.expander("Optim JSON", expanded=False):
        st.write(optim)

    cycle = st.selectbox(
        "Select a cycle",
        list(range(len(optim))),
        format_func=lambda i: f"Cycle {i + 1}",
    )
    losses = optim[cycle]["losses"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(losses["obs_losses"]))),
            y=losses["obs_losses"],
            name="Observation loss",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(losses["dynamical_reg"]))),
            y=losses["dynamical_reg"],
            name="Dynamical regularization",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(losses["background_reg"]))),
            y=losses["background_reg"],
            name="Background regularization",
        )
    )

    fig.update_layout(
        title="Training loss evolution", xaxis={"title": "Epochs"}
    )

    st.plotly_chart(fig)
