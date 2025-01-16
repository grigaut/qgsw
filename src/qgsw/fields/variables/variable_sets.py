"""Create specific variable sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.coefficients import LSRSFInferredAlpha
from qgsw.fields.variables.dynamics import (
    CollinearityCoefficientDiag,
    Enstrophy,
    LayerDepthAnomalyDiag,
    MeridionalVelocityDiag,
    MeridionalVelocityFlux,
    PhysicalLayerDepthAnomaly,
    PhysicalMeridionalVelocity,
    PhysicalSurfaceHeightAnomaly,
    PhysicalVorticity,
    PhysicalZonalVelocity,
    PotentialVorticity,
    Pressure,
    PressureTilde,
    StreamFunction,
    SurfaceHeightAnomaly,
    TimeDiag,
    TotalEnstrophy,
    Vorticity,
    ZonalVelocityDiag,
    ZonalVelocityFlux,
)
from qgsw.fields.variables.energetics import (
    ModalAvailablePotentialEnergy,
    ModalEnergy,
    ModalKineticEnergy,
    TotalAvailablePotentialEnergy,
    TotalEnergy,
    TotalKineticEnergy,
)
from qgsw.masks import Masks
from qgsw.models.qg.collinear_sublayer.core import QGCollinearSF
from qgsw.models.qg.collinear_sublayer.stretching_matrix import (
    compute_A_collinear_sf,
)
from qgsw.models.qg.core import QG
from qgsw.models.qg.stretching_matrix import compute_A

if TYPE_CHECKING:
    import torch

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable


def _qg_variable_set(
    physics_config: PhysicsConfig,
    space_config: SpaceConfig,
    model_config: ModelConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, DiagnosticVariable]:
    """Prepare QG variable set.

    Args:
        run_output (RunOutput): Run output.
        physics_config (PhysicsConfig): Physics configuration
        space_config (SpaceConfig): Space configuration
        model_config (ModelConfig): Model configuration
        dtype (torch.dtype): Data type
        device (torch.device): Device
    """
    dx = space_config.dx
    dy = space_config.dy
    ds = space_config.ds
    H = model_config.h  # noqa: N806
    g_prime = model_config.g_prime
    A = compute_A(  # noqa: N806
        H,
        g_prime,
        dtype,
        device,
    )
    t = TimeDiag()
    u = ZonalVelocityDiag()
    v = MeridionalVelocityDiag()
    h = LayerDepthAnomalyDiag()
    u_phys = PhysicalZonalVelocity(dx)
    v_phys = PhysicalMeridionalVelocity(dy)
    h_phys = PhysicalLayerDepthAnomaly(ds)
    U = ZonalVelocityFlux(dx)  # noqa: N806
    V = MeridionalVelocityFlux(dy)  # noqa: N806
    vorticity = Vorticity(
        Masks.empty(space_config.nx, space_config.ny, device),
        slip_coef=physics_config.slip_coef,
    )
    vorticity_phys = PhysicalVorticity(vorticity, ds)
    eta = SurfaceHeightAnomaly()
    eta_phys = PhysicalSurfaceHeightAnomaly(h_phys)
    p = Pressure(g_prime.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), eta_phys)
    psi = StreamFunction(p, physics_config.f0)
    pv = PotentialVorticity(
        vorticity_phys,
        H.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * ds,
        ds,
        physics_config.f0,
    )
    enstrophy = Enstrophy(pv)
    enstrophy_tot = TotalEnstrophy(pv)
    ke_hat = ModalKineticEnergy(A, psi, H, dx, dy)
    ape_hat = ModalAvailablePotentialEnergy(A, psi, H, physics_config.f0)
    energy_hat = ModalEnergy(ke_hat, ape_hat)
    ke = TotalKineticEnergy(psi, H, dx, dy)
    ape = TotalAvailablePotentialEnergy(A, psi, H, physics_config.f0)
    energy = TotalEnergy(ke, ape)
    lsr_sf_alpha = LSRSFInferredAlpha(psi)

    return {
        t.name: t,
        u.name: u,
        v.name: v,
        h.name: h,
        u_phys.name: u_phys,
        v_phys.name: v_phys,
        h_phys.name: h_phys,
        U.name: U,
        V.name: V,
        vorticity.name: vorticity,
        vorticity_phys.name: vorticity_phys,
        enstrophy.name: enstrophy,
        enstrophy_tot.name: enstrophy_tot,
        eta: eta,
        eta_phys.name: eta_phys,
        p.name: p,
        psi.name: psi,
        pv.name: pv,
        ke_hat.name: ke_hat,
        ape_hat.name: ape_hat,
        energy_hat.name: energy_hat,
        ke.name: ke,
        ape.name: ape,
        energy.name: energy,
        lsr_sf_alpha.name: lsr_sf_alpha,
    }


def _collinear_qg_variable_set(
    physics_config: PhysicsConfig,
    space_config: SpaceConfig,
    model_config: ModelConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, DiagnosticVariable]:
    """Prepare Collinear QG variable set.

    Args:
        run_output (RunOutput): Run output.
        physics_config (PhysicsConfig): Physics configuration
        space_config (SpaceConfig): Space configuration
        model_config (ModelConfig): Model configuration
        dtype (torch.dtype): Data type
        device (torch.device): Device
    """
    dx = space_config.dx
    dy = space_config.dy
    ds = space_config.ds
    H = model_config.h[:1]  # noqa: N806
    if model_config.collinearity_coef.type == "alpha_constant":
        alpha = model_config.collinearity_coef.value
    elif model_config.collinearity_coef.type == "alpha_lsr_sf":
        alpha = model_config.collinearity_coef.initial
    A = compute_A_collinear_sf(  # noqa: N806
        model_config.h,
        model_config.g_prime,
        alpha,
        dtype,
        device,
    )
    t = TimeDiag()
    alpha = CollinearityCoefficientDiag()
    u = ZonalVelocityDiag()
    v = MeridionalVelocityDiag()
    h = LayerDepthAnomalyDiag()
    u_phys = PhysicalZonalVelocity(dx)
    v_phys = PhysicalMeridionalVelocity(dy)
    h_phys = PhysicalLayerDepthAnomaly(ds)
    U = ZonalVelocityFlux(dx)  # noqa: N806
    V = MeridionalVelocityFlux(dy)  # noqa: N806
    vorticity = Vorticity(
        Masks.empty(space_config.nx, space_config.ny, device),
        slip_coef=physics_config.slip_coef,
    )
    vorticity_phys = PhysicalVorticity(vorticity, ds)
    eta = SurfaceHeightAnomaly()
    eta_phys = PhysicalSurfaceHeightAnomaly(h_phys)
    p = PressureTilde(
        model_config.g_prime.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
        eta_phys,
    )
    psi = StreamFunction(p, physics_config.f0)
    pv = PotentialVorticity(
        vorticity_phys,
        H[:1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * ds,
        ds,
        physics_config.f0,
    )
    ape = TotalAvailablePotentialEnergy(A, psi, H, physics_config.f0)
    ke = TotalKineticEnergy(psi, H, dx, dy)
    ke_hat = ModalKineticEnergy(A, psi, H, dx, dy)
    ape_hat = ModalAvailablePotentialEnergy(A, psi, H, physics_config.f0)

    enstrophy = Enstrophy(pv)
    enstrophy_tot = TotalEnstrophy(pv)
    energy_hat = ModalEnergy(ke_hat, ape_hat)
    energy = TotalEnergy(ke, ape)
    return {
        t.name: t,
        alpha.name: alpha,
        u.name: u,
        v.name: v,
        h.name: h,
        u_phys.name: u_phys,
        v_phys.name: v_phys,
        h_phys.name: h_phys,
        U.name: U,
        V.name: V,
        vorticity.name: vorticity,
        vorticity_phys.name: vorticity_phys,
        enstrophy.name: enstrophy,
        enstrophy_tot.name: enstrophy_tot,
        eta: eta,
        eta_phys.name: eta_phys,
        p.name: p,
        psi.name: psi,
        pv.name: pv,
        ke_hat.name: ke_hat,
        ape_hat.name: ape_hat,
        energy_hat.name: energy_hat,
        ke.name: ke,
        ape.name: ape,
        energy.name: energy,
    }


def create_qg_variable_set(
    physics_config: PhysicsConfig,
    space_config: SpaceConfig,
    model_config: ModelConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, DiagnosticVariable]:
    """Prepare QG variable set.

    Args:
        run_output (RunOutput): Run output.
        physics_config (PhysicsConfig): Physics configuration
        space_config (SpaceConfig): Space configuration
        model_config (ModelConfig): Model configuration
        dtype (torch.dtype): Data type
        device (torch.device): Device
    """
    if model_config.type == QG.get_type():
        return _qg_variable_set(
            physics_config=physics_config,
            space_config=space_config,
            model_config=model_config,
            dtype=dtype,
            device=device,
        )
    if model_config.type == QGCollinearSF.get_type():
        return _collinear_qg_variable_set(
            physics_config=physics_config,
            space_config=space_config,
            model_config=model_config,
            dtype=dtype,
            device=device,
        )
    msg = (
        "Supported model types are `QG` and `QGCollinearSF`"
        f", received {model_config.type}"
    )
    raise ValueError(msg)
