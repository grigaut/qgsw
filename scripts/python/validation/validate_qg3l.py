"""Validate QG3L results."""

import gc
import sys
from collections.abc import Callable

import torch

from qgsw.cli import parse_output_folder, parse_run_id
from qgsw.configs.core import Configuration
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.core import build_basis_from_params_dict
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.logging.core import getLogger, setup_root_logger
from qgsw.logging.utils import box, step
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.sv.core import SurfaceVorticity
from qgsw.scripts.boundaries import extract_psi_bc, extract_wide_psi_bc
from qgsw.scripts.model_interactions import init_model
from qgsw.scripts.output import (
    OutputFolder,
    get_tracking_filename,
    retrieve_output_folder,
)
from qgsw.scripts.pv import build_pv_funcs
from qgsw.scripts.setups.model import setup_model
from qgsw.scripts.validation import Losses
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import grad_perp
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
)
from qgsw.specs import defaults, from_tensor
from qgsw.utils import covphys
from qgsw.utils.reshaping import crop

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

logger = getLogger(__name__)

specs = defaults.get()

dt = 7200
n_steps_per_cycle = 250

b = 4


def setup(
    cycle: int,
    output: OutputFolder,
    model: QGPSIQ | SurfaceVorticity,
    sfs: torch.Tensor,
    times: list[torch.Tensor],
    build_pv: Callable[
        [torch.Tensor, torch.Tensor],
        tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[Boundaries], Boundaries],
        ],
    ],
) -> bool:
    """Setup model.

    Args:
        cycle (int): Cycle.
        output (OutputFolder): Output.
        model (QGPSIQ | SurfaceVorticity): Model.
        sfs (torch.Tensor): Stream functions.
        times (list[torch.Tensor]): Times.
        build_pv (Callable[ [torch.Tensor, torch.Tensor], tuple[ Callable[[torch.Tensor], torch.Tensor], Callable[[Boundaries], Boundaries], ], ]): PV func.

    Returns:
        bool: If it has no data.
    """  # noqa: E501
    nb_cycles = len(output.optim.load())
    if cycle >= nb_cycles:
        return True
    bases_params = output.basis.load()[cycle]
    results = output.results.load()[cycle]
    if model.get_type() == QGPSIQForced.get_type():
        basis_external = build_basis_from_params_dict(bases_params["external"])
        coefs = DecompositionCoefs.from_dict(results["external"])
        basis_external.set_coefs(coefs)
        model.basis_external = basis_external
    elif model.get_type() == QGPSIQRGPsi2TransportDR.get_type():
        try:
            alpha = results["alpha"]
        except KeyError:
            alpha = torch.tensor(0, **from_tensor(model.A))
        model.alpha = alpha
        basis_dynamical = build_basis_from_params_dict(
            bases_params["dynamical"]
        )
        coefs = DecompositionCoefs.from_dict(results["dynamical"])
        basis_dynamical.set_coefs(coefs)
        model.basis_dynamical = basis_dynamical
    elif model.get_type() == QGPSIQForcedSurfML.get_type():
        try:
            alpha = results["alpha"]
        except KeyError:
            alpha = torch.tensor(0, **from_tensor(model.A))
        model.alpha = alpha
        basis_dynamical = build_basis_from_params_dict(
            bases_params["dynamical"]
        )
        coefs = DecompositionCoefs.from_dict(results["dynamical"])
        basis_dynamical.set_coefs(coefs)
        model.basis_dynamical = basis_dynamical
        basis_external = build_basis_from_params_dict(bases_params["external"])
        coefs = DecompositionCoefs.from_dict(results["external"])
        basis_external.set_coefs(coefs)
        model.basis_external = basis_external
    elif model.get_type() == SurfaceVorticity.get_type():
        basis_psi_a = build_basis_from_params_dict(bases_params["psi"])
        coefs = DecompositionCoefs.from_dict(results["psi"])
        basis_psi_a.set_coefs(coefs)
        model.basis_psi_a = basis_psi_a
        basis_chi_a = build_basis_from_params_dict(bases_params["chi"])
        coefs = DecompositionCoefs.from_dict(results["chi"])
        basis_chi_a.set_coefs(coefs)
        model.basis_chi_a = basis_chi_a
    psi0 = sfs[0]
    psi_bcs = [extract_psi_bc(psi, b) for psi in sfs]
    psi_bcs_w = [extract_wide_psi_bc(psi, b, clone=True) for psi in sfs]
    init_model(model, psi0, psi_bcs, psi_bcs_w, build_pv, times, b)
    return False


if __name__ == "__main__":
    output_folder = parse_output_folder()
    run_id = parse_run_id()

    try:
        output = retrieve_output_folder(
            output_folder,
            run_id,
        )
    except FileNotFoundError:
        msg = "ERROR: Not output file available."
        logger.critical(msg)
        sys.exit(1)

    params = output.params.load()
    setup_root_logger(params["verbose"])

    config = Configuration.from_toml(output.config)
    n_cycles: int = params["cycles"]

    ## Areas

    indices: tuple[int, int, int, int] = params["indices"]
    imin, imax, jmin, jmax = indices
    psi_slices_w = [
        slice(imin - b, imax + b + 1),
        slice(jmin - b, jmax + b + 1),
    ]

    def extract_psi_w(psi: torch.Tensor) -> torch.Tensor:
        """Extract psi."""
        return psi[..., psi_slices_w[0], psi_slices_w[1]]

    space = SpaceDiscretization3D.from_config(
        config.space,
        config.model,
    )

    space_slice = space.remove_h().slice(imin, imax + 1, jmin, jmax + 1)

    space_slice_wide = space.remove_h().slice(
        imin - b, imax + b + 1, jmin - b, jmax + b + 1
    )

    separation = int(params["separation"] * dt / 3600 / 24)

    # Parameters

    H = config.model.h
    g_prime = config.model.g_prime
    beta_plane = config.physics.beta_plane
    bottom_drag_coef = config.physics.bottom_drag_coefficient
    slip_coef = config.physics.slip_coef

    P = QGProjector(
        A=compute_A(H=H, g_prime=g_prime),
        H=H.unsqueeze(-1).unsqueeze(-1),
        space=space,
        f0=beta_plane.f0,
        masks=Masks.empty(
            nx=config.space.nx,
            ny=config.space.ny,
        ),
    )
    dx, dy = space.dx, space.dy

    wind = WindForcing.from_config(
        config.windstress,
        config.space,
        config.physics,
    )
    tx, ty = wind.compute()

    uvh0 = UVH.from_file(config.simulation.startup.file)

    U: float = (
        (uvh0.u[0, 0].max() ** 2 + uvh0.v[0, 0].max() ** 2).sqrt().item()
    )
    L: float = dx.item()
    T: float = L / U

    psi_start = P.compute_p(covphys.to_cov(uvh0, dx, dy))[0] / beta_plane.f0

    # Reference model

    model_3l = QGPSIQ(
        space_2d=space.remove_h(),
        H=H,
        beta_plane=config.physics.beta_plane,
        g_prime=g_prime,
    )
    model_3l.set_wind_forcing(tx, ty)
    model_3l.bottom_drag_coef = bottom_drag_coef
    model_3l.slip_coef = slip_coef
    model_3l.dt = dt
    y0 = model_3l.y0

    model_3l.reset_time()
    model_3l.set_psi(psi_start)

    y_wide = space_slice_wide.q.xy.y[0, :].unsqueeze(0)
    beta_effect_wide = beta_plane.beta * (y_wide - y0)

    if params["perturbed_params"]:
        msg = "Using perturbed params."
        logger.info(box(msg))
        H1_, H2_ = 150, 1350
        g1_, g2_ = g_prime[0], g_prime[1] * 1
    else:
        H1_, H2_ = H[0], H[1]
        g1_, g2_ = g_prime[0], g_prime[1]

    # Model
    model = setup_model(
        params["model_type"],
        space_slice,
        torch.tensor([H1_, H2_], **specs),
        torch.tensor([g1_, g2_], **specs),
        beta_plane,
        dt,
        config.physics.slip_coef,
    )
    model.y0 = y0
    if not params["no_wind"]:
        model.set_wind_forcing(
            tx[imin:imax, jmin : jmax + 1], ty[imin : imax + 1, jmin:jmax]
        )
    # PV computation
    build_q_rg = build_pv_funcs(
        dx,
        dy,
        beta_plane.f0,
        beta_effect_wide
        * (params["model_type"] != SurfaceVorticity.get_type()),
        b,
        space_slice_wide.nx,
    )

    outputs = []

    for c in range(n_cycles):
        msg = f"Computing loss for cycle {step(c + 1, n_cycles)}"
        logger.info(box(msg))
        losses = Losses(space_slice)
        losses.with_psi2()
        losses.with_uv()
        losses.cycle = c
        torch.cuda.reset_peak_memory_stats()

        imin, imax, jmin, jmax = indices

        psis = [extract_psi_w(model_3l.psi[:, :2])]
        times = [model_3l.time.item()]

        for _ in range(1, n_steps_per_cycle):
            model_3l.step()
            psis.append(extract_psi_w(model_3l.psi[:, :2]))
            times.append(model_3l.time.item())

        psis_ref = torch.stack(psis, dim=0)

        losses.no_data = setup(
            c, output, model, psis_ref[:, :, :1, ...], times, build_q_rg
        )

        model.reset_time()

        u_refs, v_refs = grad_perp(psis_ref)
        u_refs /= dy
        v_refs /= dx

        losses.compute_loss(model, crop(psis_ref[0], b))
        losses.compute_psi2_losses(model, crop(psis_ref[0], b))
        losses.compute_uv_losses(model, crop(u_refs[0], b), crop(v_refs[0], b))
        for n in range(1, n_steps_per_cycle):
            if not losses.no_data:
                model.step()
            losses.compute_loss(model, crop(psis_ref[n], b))
            losses.compute_psi2_losses(model, crop(psis_ref[n], b))
            losses.compute_uv_losses(
                model, crop(u_refs[n], b), crop(v_refs[n], b)
            )

        torch.cuda.empty_cache()
        gc.collect()

        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        msg_mem = (
            f"Cycle {step(c + 1, n_cycles)} | "
            f"Max memory allocated: {max_mem:.1f} MB."
        )
        logger.info(box(msg_mem, style="="))

        outputs.append(losses.losses)
        output.scores.save(outputs)

        for _ in range(separation):
            model_3l.step()
    if run_id is not None:
        tracking_file = get_tracking_filename(run_id)
        tracking_file.unlink(missing_ok=True)
        msg = "Tracking file removed."
        logger.info(box(msg))
