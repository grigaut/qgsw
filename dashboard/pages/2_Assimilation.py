"""Assimilation plot."""

from pathlib import Path

import streamlit as st
import torch

from qgsw.fields.errors.error_sets import create_errors_set
from qgsw.fields.scope import Scope
from qgsw.fields.variables.utils import check_unit_compatibility
from qgsw.fields.variables.variable_sets import create_qg_variable_set
from qgsw.output import RunOutput
from qgsw.plots.heatmaps import AnimatedHeatmaps
from qgsw.plots.scatter import ScatterPlot
from qgsw.specs import DEVICE

ROOT = Path(__file__).parent.parent.parent
OUTPUTS = ROOT.joinpath("output")

local_sources = list(OUTPUTS.glob("local/*/_summary.toml"))
g5k_sources = list(OUTPUTS.glob("g5k/*/_summary.toml"))
sources = [file.parent for file in local_sources + g5k_sources]

st.title("Assimilation")

folder = st.selectbox("Data source", options=sources)

run = RunOutput(folder)

if run.summary.configuration.simulation.kind != "assimilation":
    st.error("The selected simulation is not an assimilation one.", icon="⚠️")
    st.stop()

config = run.summary.configuration

errors = create_errors_set()

vars_dict = create_qg_variable_set(
    config.physics,
    config.space,
    config.model,
    torch.float64,
    DEVICE.get(),
)
st.write(run)

prefix_ref = run.summary.configuration.simulation.reference.prefix
run_ref = RunOutput(folder, prefix=prefix_ref)

vars_dict_ref = create_qg_variable_set(
    config.physics,
    config.space,
    config.simulation.reference,
    torch.float64,
    DEVICE.get(),
)
levels_nb = run.summary.configuration.model.h.shape[0]

if run.summary.configuration.model.type == "QGCollinearSF":
    levels_nb -= 1

if not run.summary.is_finished:
    st.warning("The simulation did not reach its end.", icon="⚠️")


st.title("Point-wise variables")

show_error_pts = st.toggle("Display Errors", key="toggle-pts")

if show_error_pts:
    error_type_pts = st.selectbox(
        "Error to display",
        [e for e in errors.values() if e.get_scope() == Scope.POINT_WISE],
        format_func=lambda x: x.get_description(),
        key="sbox-pts",
    )

vars_pts = [var for var in vars_dict.values() if var.scope == Scope.POINT_WISE]

selected_var_pts = st.selectbox("Variable to display", vars_pts)
selected_var_pts_ref = vars_dict_ref[selected_var_pts.name]

if show_error_pts:
    error_pts = error_type_pts(selected_var_pts, selected_var_pts_ref)
    error_pts.slice = [slice(None, None), slice(0, 1), ...]

with st.form(key="var-form"):
    level = st.selectbox("Level", list(range(levels_nb)))
    submit_pts = st.form_submit_button("Display")

if submit_pts:
    uvhs_ref = (output.read() for output in run_ref.outputs())
    uvhs = (output.read() for output in run.outputs())
    if not show_error_pts:
        values_ref = (selected_var_pts_ref.compute(uvh) for uvh in uvhs_ref)
        data_ref = [d[0, level].T.cpu() for d in values_ref]
        values = (selected_var_pts.compute(uvh) for uvh in uvhs)
        data = [d[0, level].T.cpu() for d in values]

        plot_pts = AnimatedHeatmaps(
            [data_ref, data],
        )
        plot_pts.set_subplot_titles(
            [f"Reference - Level {level}", f"Model - Level {level}"],
        )
    else:
        data = [
            error_pts.compute_point_wise(uvh, uvh_ref)[0, level].T.cpu()
            for uvh, uvh_ref in zip(uvhs, uvhs_ref)
        ]
        plot_pts = AnimatedHeatmaps(
            [data],
        )
        plot_pts.set_subplot_titles(
            [f"Error - Level {level}"],
        )
    plot_pts.set_colorbar_text(
        f"{selected_var_pts.description} [{selected_var_pts.unit.value}]",
    )
    plot_pts.set_slider_prefix("Time: ")
    plot_pts.set_figure_size(800, 1000)
    st.plotly_chart(plot_pts.retrieve_figure(), use_container_width=True)

st.title("Level-wise variables")

show_error_lvl = st.toggle("Display Errors", key="toggle-lvl")

if show_error_lvl:
    error_type_lvl = st.selectbox(
        "Error to display",
        [
            e
            for e in errors.values()
            if e.get_scope() in [Scope.POINT_WISE, Scope.LEVEL_WISE]
        ],
        format_func=lambda x: x.get_description(),
        key="sbox-lvl",
    )

if show_error_lvl:
    vars_lvl = [
        v for v in vars_dict.values() if v.scope == error_type_lvl.get_scope()
    ]
else:
    vars_lvl = [v for v in vars_dict.values() if v.scope == Scope.LEVEL_WISE]

selected_vars_lvl = st.multiselect("Variable to display", vars_lvl, key="lvl")
selected_vars_lvl_ref = [vars_dict_ref[v.name] for v in selected_vars_lvl]

if not check_unit_compatibility(*selected_vars_lvl):
    st.error("Selected variables don't have the same unit.")
    st.stop()

if show_error_lvl:
    errors_lvl = [
        error_type_lvl(var, var_ref)
        for var, var_ref in zip(selected_vars_lvl, selected_vars_lvl_ref)
    ]
    for error in errors_lvl:
        error.slice = [slice(None, None), slice(0, 1), ...]

with st.form(key="var-form-level-wise"):
    levels = list(range(levels_nb))
    selected_lvl = st.multiselect("Level(s)", levels)
    submit_lvl = st.form_submit_button("Display")


if submit_lvl:
    if not show_error_lvl:
        datas = [
            [
                d[0, level].cpu()
                for d in (
                    var.compute(uvh)
                    for uvh in (output.read() for output in run.outputs())
                )
            ]
            for level in selected_lvl
            for var in selected_vars_lvl
        ]
        names = [
            f"{var.description} - Ens: 0 - Level: {level}"
            for level in selected_lvl
            for var in selected_vars_lvl
        ]
        datas_ref = [
            [
                d[0, level].cpu()
                for d in (
                    var.compute(uvh)
                    for uvh in (output.read() for output in run_ref.outputs())
                )
            ]
            for level in selected_lvl
            for var in selected_vars_lvl_ref
        ]
        names_ref = [
            f"{var.description} - Ens: 0 - Level: {level} (reference)"
            for level in selected_lvl
            for var in selected_vars_lvl
        ]

        plot_lvl = ScatterPlot(datas=datas_ref + datas)
        plot_lvl.set_traces_name(*names_ref, *names)
        xs = [
            list(run.seconds())
            for _ in selected_lvl
            for _ in selected_vars_lvl
        ]
        xs_ref = [
            list(run_ref.seconds()) for _ in levels for _ in selected_vars_lvl
        ]
        plot_lvl.set_xs(*xs, *xs_ref)
    else:
        uvhs = [
            (output.read() for output in run.outputs())
            for _ in selected_lvl
            for _ in selected_vars_lvl
        ]
        uvhs_ref = [
            (output.read() for output in run_ref.outputs())
            for _ in selected_lvl
            for _ in selected_vars_lvl
        ]
        datas = [
            [
                errors_lvl[k].compute_level_wise(uvh, uvh_ref)[0, lvl].cpu()
                for uvh, uvh_ref in zip(uvhs[k], uvhs_ref[k])
            ]
            for lvl in selected_lvl
            for k in range(len(selected_vars_lvl))
        ]
        names = [
            f"{var.description} - Ens: 0 - Level: {level}"
            for level in selected_lvl
            for var in selected_vars_lvl
        ]
        plot_lvl = ScatterPlot(datas)
        plot_lvl.set_traces_name(*names)
        xs = [
            list(run.seconds())
            for _ in selected_lvl
            for _ in selected_vars_lvl
        ]
        plot_lvl.set_xs(*xs)
    plot_lvl.set_xaxis_title("Time [s]")
    plot_lvl.set_yaxis_title(f"[{selected_vars_lvl[0].unit.value}]")
    plot_lvl.set_figure_size(height=750, width=500)
    st.plotly_chart(plot_lvl.retrieve_figure(), use_container_width=True)

st.title("Ensemble-wise variables")

show_error_ens = st.toggle("Display Errors", key="toggle-ens")

if show_error_ens:
    error_type_ens = st.selectbox(
        "Error to display",
        list(errors.values()),
        format_func=lambda x: x.get_description(),
        key="sbox-ens",
    )

if show_error_ens:
    vars_ens = [
        v for v in vars_dict.values() if v.scope == error_type_ens.get_scope()
    ]
else:
    vars_ens = [
        v for v in vars_dict.values() if v.scope == Scope.ENSEMBLE_WISE
    ]

selected_vars_ens = st.multiselect("Variable to display", vars_ens, key="ens")
selected_vars_ens_ref = [vars_dict_ref[v.name] for v in selected_vars_ens]

if not check_unit_compatibility(*selected_vars_ens):
    st.error("Selected variables don't have the same unit.")
    st.stop()

if show_error_ens:
    errors_ens = [
        error_type_ens(var, var_ref)
        for var, var_ref in zip(selected_vars_ens, selected_vars_ens_ref)
    ]

with st.form(key="var-form-ensemble-wise"):
    submit_ensemble = st.form_submit_button("Display")

if submit_ensemble:
    if not show_error_ens:
        datas = [
            [
                d[0].cpu()
                for d in (
                    var.compute(uvh)
                    for uvh in (output.read() for output in run.outputs())
                )
            ]
            for var in selected_vars_ens
        ]
        names = [f"{var.description} - Ens: 0" for var in selected_vars_ens]
        datas_ref = [
            [
                d[0].cpu()
                for d in (
                    var.compute(uvh)
                    for uvh in (output.read() for output in run_ref.outputs())
                )
            ]
            for var in selected_vars_ens_ref
        ]
        names_ref = [
            f"{var.description} - Ens: 0 (reference)"
            for var in selected_vars_ens
        ]

        plot_ens = ScatterPlot(datas=datas_ref + datas)
        plot_ens.set_traces_name(*names_ref, *names)
        xs = [list(run.seconds()) for _ in selected_vars_ens]
        xs_ref = [list(run_ref.seconds()) for _ in selected_vars_ens]
        plot_ens.set_xs(*xs, *xs_ref)
    else:
        uvhs = [
            (output.read() for output in run.outputs())
            for _ in selected_vars_ens
        ]
        uvhs_ref = [
            (output.read() for output in run_ref.outputs())
            for _ in selected_vars_ens
        ]
        datas = [
            [
                errors_ens[k].compute_ensemble_wise(uvh, uvh_ref)[0].cpu()
                for uvh, uvh_ref in zip(uvhs[k], uvhs_ref[k])
            ]
            for k in range(len(selected_vars_ens))
        ]
        names = [f"{var.description} - Ens: 0" for var in selected_vars_ens]
        plot_ens = ScatterPlot(datas)
        plot_ens.set_traces_name(*names)
        xs = [list(run.seconds()) for _ in selected_vars_ens]
        xs_ref = [
            list(run_ref.seconds()) for _ in levels for _ in selected_vars_ens
        ]
        plot_ens.set_xs(*xs, *xs_ref)
    plot_ens.set_xaxis_title("Time [s]")
    plot_ens.set_yaxis_title(
        f"[{selected_vars_ens[0].unit.value}]",
    )
    plot_ens.set_figure_size(height=750, width=500)
    st.plotly_chart(plot_ens.retrieve_figure(), use_container_width=True)
