"""Browser results files."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytz
import seaborn as sns
import streamlit as st
from st_copy import copy_button

from qgsw.logging.utils import sec2text
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

prefixes = [m.name[:-25] for m in matching_folders]

selected_prefixes = st.multiselect(
    "Select prefixes to narrow selection",
    np.unique(prefixes).tolist(),
    default=None,
)

dates = [
    datetime.strptime(m.name[-24:-9], "%Y%m%d_%H%M%S")
    .astimezone(pytz.utc)
    .date()
    for m in matching_folders
]

selected_dates = st.multiselect(
    "Select dates to narrow selection",
    np.unique(dates).tolist(),
    default=None,
)

if selected_prefixes:
    prefixes_folders = [p in selected_prefixes for p in prefixes]
else:
    prefixes_folders = [True for d in prefixes]
if selected_dates:
    dates_folders = [d in selected_dates for d in dates]
else:
    dates_folders = [True for d in dates]
folder = st.selectbox(
    "Select a folder",
    [
        f
        for f, d, p in zip(matching_folders, dates_folders, prefixes_folders)
        if d and p
    ],
    index=None,
    format_func=lambda p: p.name,
)


if folder is not None:
    col1, col2 = st.columns([0.90, 0.1])

    col1.header(f"Monitoring {folder.name}", anchor=False)
    with col2:
        st.markdown(" ")
        st.markdown(" ")
        copy_button(
            folder.name,
            tooltip="Copy folder name",
            copied_label=f"Copied {folder.name}",
            icon="st",
        )

    date = datetime.strptime(folder.name[-24:-9], "%Y%m%d_%H%M%S").replace(
        tzinfo=pytz.UTC
    )
    st.subheader(date.strftime("%A %B %d %G - %X"), anchor=False)

    output_folder = build_output_folder(folder)

    if not output_folder.files_exist():
        st.error("Output folder files don't exist")
        st.stop()

    st.divider()
    ### Parameters

    st.subheader("Parameters")
    params: dict[str, Any] = output_folder.params.load()

    badges = []

    badges.append(f":green-badge[optimizer: {params['optimizer']}]")

    if params["obs_track"]:
        badges.append(":blue-badge[Observations sampled along tracks]")
    else:
        badges.append(":blue-badge[Entire surface observed]")

    badges.append(f":violet-badge[{params['cycles']} cycles]")

    if params["no_reg"]:
        badges.append(":orange-badge[No regularization]")
    else:
        badges.append(":green-badge[With regularization]")

    if params.get("perturbed_params"):
        badges.append(":orange-badge[Perturbed params]")

    if "no_wind" in params:
        no_wind = params["no_wind"]
        if no_wind:
            badges.append(":orange-badge[Without wind]")
        else:
            badges.append(":green-badge[With wind]")
            if "wind_optim" in params:
                if params["wind_optim"]:
                    badges.append(":green-badge[With wind optimization]")
                else:
                    badges.append(":orange-badge[Without wind optimization]")

    if "indices" in params:
        imin, imax, jmin, jmax = params["indices"]
        badges.append(
            f":violet-badge[Area: [{imin}, {imax}] x [{jmin}, {jmax}]]"
        )
    if "season" in params:
        badges.append(f":violet-badge[Season: {params['season']}]")

    if badges:
        st.markdown(" ".join(badges))

    model_type = params["model_type"]
    st.header(f"Model type: {model_type}", text_alignment="center")

    st.header("Bases")

    bases_names = ["dyn", "ext", "psi", "chi"]

    tabs = st.columns(len(bases_names), border=True)

    for name, tab in zip(bases_names, tabs):
        with tab:
            basis = (
                params[f"basis_{name}"],
                params[f"basis_{name}_lr"],
                params[f"basis_{name}_bkg_reg"],
                params[f"basis_{name}_no_optim"],
            )
            st.header(f"Basis '{name}'")
            st.metric("Basis type", basis[0])
            st.metric("Learning rate", basis[1])
            st.metric(
                "Background reg",
                f":{'green' if basis[2] else 'red'}[{basis[2]}]",
            )
            with_optim = not basis[3]
            st.metric(
                "With optim",
                f":{'green' if with_optim else 'red'}[{with_optim}]",
            )

    reg_names = ["qg", "div"]

    st.header("Dynamical regularizations")

    tabs = st.columns(len(reg_names), border=True)

    for name, tab in zip(reg_names, tabs):
        with tab:
            reg = (
                params[f"dyn_reg_{name}"],
                params[f"gamma_dyn_{name}"],
            )
            st.header(f"Regularization type '{name}'")
            st.metric(
                "With regularization",
                f":{'green' if reg[0] else 'red'}[{reg[0]}]",
            )
            st.metric("Gamma", reg[1])

    with st.expander("Parameters JSON", expanded=False):
        st.write(params)

    st.divider()
    ### Optimization results

    st.subheader("Optimization results")
    optim = output_folder.optim.load()

    cycle = st.selectbox(
        "Select a cycle",
        list(range(len(optim))),
        format_func=lambda i: f"Cycle {i + 1}",
    )
    opt_cycle: dict = optim[cycle]
    losses = opt_cycle["losses"]

    n_optim = opt_cycle["max_steps"]
    try:
        overflow = opt_cycle["overflow"]
    except KeyError:
        overflow = False

    if (n_optim > (conv := opt_cycle["nb_steps"])) and (not overflow):
        st.badge(
            f"Convergence reached in {conv} epochs",
            icon=":material/check:",
            color="green",
        )
    elif (n_optim != opt_cycle["nb_steps"]) and overflow:
        st.badge(
            "Overflow during minimization",
            icon=":material/error:",
            color="red",
        )
    else:
        st.badge(
            f"No convergence reached after {n_optim} epochs", color="blue"
        )

    fig = go.Figure()

    losses_arr = {
        k: np.array(
            v
        )  # np.concatenate([np.array(v), np.ones(n_optim - len(v)) * np.nan])
        for k, v in losses.items()
    }

    loss_names = {
        "obs": "Observation loss",
        "dyn": "Dynamical regularization",
        "bkg": "Background regularization",
        "best": "Best loss",
    }

    loss_tot = losses_arr["obs"] + losses_arr["bkg"] + losses_arr["dyn"]

    fig.add_trace(
        go.Scatter(
            x=list(range(len(loss_tot))),
            y=loss_tot,
            name="Total loss",
        )
    )

    for key, loss in losses_arr.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(loss))),
                y=loss,
                name=loss_names[key],
            )
        )

    fig.update_layout(
        title="Training loss evolution", xaxis={"title": "Epochs"}
    )

    col1, col2, col3, col4 = st.columns(4, border=True)

    argmin_loss = np.nanargmin(loss_tot)

    best_loss = loss_tot[argmin_loss]
    best_obs_loss = losses_arr["obs"][argmin_loss]
    best_dyn_reg = losses_arr["dyn"][argmin_loss]
    best_bkg_reg = losses_arr["bkg"][argmin_loss]

    col1.metric(
        "Final loss",
        best_loss,
        format="%06.4f",
        height="stretch",
        delta_color="blue",
        delta=f"{1 - best_loss / loss_tot[0]:.2%}",
        delta_description="reduction of initial loss",
        delta_arrow="off",
        chart_data=(1 - loss_tot / loss_tot[0]) * 100,
        chart_type="area",
    )
    col2.metric(
        "Final observation loss",
        best_obs_loss,
        f"{best_obs_loss / best_loss:.0%}",
        format="%06.4f",
        delta_arrow="off",
        delta_color="blue",
        chart_data=np.round(losses_arr["obs"] / loss_tot * 100, 2),
        delta_description="of final loss",
        chart_type="area",
    )
    col3.metric(
        "Final background regularization",
        best_bkg_reg,
        f"{best_bkg_reg / best_loss:.0%}",
        format="%06.4f",
        delta_arrow="off",
        delta_color="off"
        if (losses_arr["bkg"][~np.isnan(losses_arr["bkg"])] == 0).all()
        else "blue",
        chart_data=np.round(losses_arr["bkg"] / loss_tot * 100, 2),
        delta_description="of final loss",
        chart_type="area",
    )
    col4.metric(
        "Final dynamical regularization",
        best_dyn_reg,
        f"{best_dyn_reg / best_loss:.0%}",
        format="%.1f",
        delta_arrow="off",
        delta_color="off"
        if (losses_arr["dyn"][~np.isnan(losses_arr["dyn"])] == 0).all()
        else "blue",
        chart_data=np.round(losses_arr["dyn"] / loss_tot * 100, 2),
        delta_description="of final loss",
        chart_type="area",
    )

    st.plotly_chart(fig)

    col1, col2 = st.columns(2, border=True)

    try:
        elapsed = opt_cycle["elapsed"]
        col1.metric(
            "Elapsed",
            sec2text(elapsed, detailed=True),
            delta=sec2text(elapsed / n_optim, detailed=True),
            delta_description="per epoch",
            delta_color="blue",
            delta_arrow="off",
        )
    except KeyError:
        col1.metric("Elapsed", None)

    col2.metric(
        "Max memory allocated",
        f"{opt_cycle['max_memory_allocated']:.0f} MB",
    )

    with st.expander("Optim JSON", expanded=False):
        st.write(optim)
