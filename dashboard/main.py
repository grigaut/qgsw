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

dates = [
    datetime.strptime(m.name[-24:-9], "%Y%m%d_%H%M%S")
    .astimezone(pytz.utc)
    .date()
    for m in matching_folders
]

selected_dates = st.multiselect(
    "Select experiment dates to narrow selection",
    np.unique(dates).tolist(),
    default=None,
)

if selected_dates:
    folders = [
        f for f, d in zip(matching_folders, dates) if d in selected_dates
    ]
else:
    folders = matching_folders
folder = st.selectbox(
    "Select a folder",
    folders,
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

    date = datetime.strptime(folder.name[-24:-9], "%Y%m%d_%H%M%S").astimezone(
        pytz.utc
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

    col1, col2, col3 = st.columns(3, border=True)

    model_type = params["model_type"]
    basis_external = params["basis_ext"]
    basis_dynamical = params["basis_dyn"]
    basis_external_lr = params["basis_ext_lr"]
    basis_dynamical_lr = params["basis_dyn_lr"]
    gamma_reg_dyn = params["gamma_reg_dyn"]
    gamma_reg_bkg = params["gamma_reg_bkg"]

    col1.metric("Model type", model_type)
    col2.metric("Basis for external forcing", basis_external)
    col3.metric("Basis for dynamical forcing", basis_dynamical)

    col2.metric("Learning rate", basis_external_lr, format="%.1e")
    col3.metric("Learning rate", basis_dynamical_lr, format="%.1e")

    if not params["no_reg"]:
        col1, col2, col3 = st.columns(3)

        col2.metric(
            "Background regularization coefficient",
            gamma_reg_bkg,
            format="%.1e",
            border=True,
        )
        col3.metric(
            "Dynamical regularizaiton coefficient",
            gamma_reg_dyn,
            format="%.1e",
            border=True,
        )

    model_type = params["model_type"]

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
    if (n_optim != (conv := opt_cycle["nb_steps"])) and (not overflow):
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
        k: np.concatenate([np.array(v), np.ones(n_optim - len(v)) * np.nan])
        for k, v in losses.items()
    }
    losses_arr["background_reg"] = gamma_reg_bkg * losses_arr["background_reg"]
    losses_arr["dynamical_reg"] = gamma_reg_dyn * losses_arr["dynamical_reg"]

    loss_names = {
        "obs_losses": "Observation loss",
        "dynamical_reg": "Dynamical regularization",
        "background_reg": "Background regularization",
    }

    loss_tot = (
        losses_arr["obs_losses"]
        + losses_arr["background_reg"]
        + losses_arr["dynamical_reg"]
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(n_optim)),
            y=loss_tot,
            name="Total loss",
        )
    )

    for key, loss in losses_arr.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(n_optim)),
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
    best_obs_loss = losses_arr["obs_losses"][argmin_loss]
    best_dyn_reg = losses_arr["dynamical_reg"][argmin_loss]
    best_bkg_reg = losses_arr["background_reg"][argmin_loss]

    col1.metric(
        "Final loss",
        best_loss,
        format="%.1f",
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
        format="%.1f",
        delta_arrow="off",
        delta_color="blue",
        chart_data=np.round(losses_arr["obs_losses"] / loss_tot * 100, 2),
        delta_description="of final loss",
        chart_type="area",
    )
    col3.metric(
        "Final background regularization",
        best_bkg_reg,
        f"{best_bkg_reg / best_loss:.0%}",
        format="%.1f",
        delta_arrow="off",
        delta_color="off"
        if (
            losses_arr["background_reg"][
                ~np.isnan(losses_arr["background_reg"])
            ]
            == 0
        ).all()
        else "blue",
        chart_data=np.round(losses_arr["background_reg"] / loss_tot * 100, 2),
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
        if (
            losses_arr["dynamical_reg"][~np.isnan(losses_arr["dynamical_reg"])]
            == 0
        ).all()
        else "blue",
        chart_data=np.round(losses_arr["dynamical_reg"] / loss_tot * 100, 2),
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
