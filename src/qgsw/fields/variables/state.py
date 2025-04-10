"""Base classes for variables."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    LayerDepthAnomaly,
    MeridionalVelocity,
    PrognosticPotentialVorticity,
    PrognosticStreamFunction,
    Time,
    ZonalVelocity,
)
from qgsw.fields.variables.prognostic_tuples import (
    PSIQ,
    PSIQT,
    UVH,
    UVHT,
    PSIQTAlpha,
    UVHTAlpha,
)

if TYPE_CHECKING:
    import torch

    from qgsw.fields.variables.base import (
        BoundDiagnosticVariable,
        PrognosticVariable,
    )


T = TypeVar("T", bound=Union[PSIQT, UVHT])


class BaseState(ABC, Generic[T]):
    """Base State class."""

    def __init__(self, prognostic: T) -> None:
        """Instantiate State.

        Args:
            prognostic (T): Prognostic variables.
        """
        self.unbind()
        self._prog = prognostic
        self._t = Time(prognostic.t)
        self._prog_vars = {Time.get_name(): self._t}

    @property
    def t(self) -> Time:
        """Time."""
        return self._t

    @t.setter
    def t(self, time: torch.Tensor) -> None:
        self.update_time(time)

    @property
    def prognostic(self) -> T:
        """Prognostic variables."""
        return self._prog

    @prognostic.setter
    def prognostic(self, prognostic: T) -> None:
        for var in self.diag_vars.values():
            var.outdated()
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    @property
    def vars(self) -> dict[str, PrognosticVariable | BoundDiagnosticVariable]:
        """List of diagnostic variables."""
        return self.prog_vars | self.diag_vars

    @property
    def diag_vars(self) -> dict[str, BoundDiagnosticVariable]:
        """Diagnostic variables."""
        return self._diag

    @property
    def prog_vars(self) -> dict[str, BoundDiagnosticVariable]:
        """Prognostic variables."""
        return self._prog_vars

    def get_repr_parts(self) -> list[str]:
        """String representations parts.

        Returns:
            list[str]: String representation parts.
        """
        txt = [
            "State",
            "└── Prognostic Variables",
        ]
        txt_prog = [f"\t├── {var}" for var in self.prog_vars.values()]
        chars = txt_prog.pop(-1).split()
        chars[0] = "\t└──"
        txt_prog.append(" ".join(chars))
        txt = txt + txt_prog
        if not self.diag_vars:
            return txt
        txt[1] = "├── Prognostic Variables"
        txt.append("└── Diagnostic Variables")
        txt_end = [f"\t├── {var}" for var in self.diag_vars.values()]
        chars = txt_end.pop(-1).split()
        chars[0] = "\t└──"
        txt_end.append(" ".join(chars))
        return txt + txt_end

    def __repr__(self) -> str:
        """String representation of State."""
        return "\n".join(self.get_repr_parts())

    def __getitem__(self, name: str) -> BoundDiagnosticVariable:
        """Get bound variables.

        Args:
            name (str): Varibale name

        Raises:
            KeyError: If the name does not correspond to a variable.

        Returns:
            BoundDiagnosticVariable: Bound variable
        """
        if name not in self.vars:
            msg = f"Variables are {', '.join(self.vars.values())}."
            raise KeyError(msg)
        return self.vars[name]

    def add_bound_diagnostic_variable(
        self,
        variable: BoundDiagnosticVariable,
    ) -> None:
        """Add a diagnostic variable.

        Args:
            variable (BoundDiagnosticVariable): Variable.
        """
        if variable.name in self.diag_vars:
            return
        self.diag_vars[variable.name] = variable

    def unbind(self) -> None:
        """Unbind all variables from state."""
        self._diag: dict[str, BoundDiagnosticVariable] = {}

    def update_time(self, time: torch.Tensor) -> None:
        """Update only the value of time.

        Args:
            time (torch.Tensor): Time.
        """
        for var in filter(lambda v: v.require_time, self.diag_vars.values()):
            var.outdated()
        prognostic = UVHT.from_uvh(
            time,
            self.prognostic.uvh,
        )
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    def increment_time(self, dt: float) -> None:
        """Increment time."""
        for var in filter(lambda v: v.require_time, self.diag_vars.values()):
            var.outdated()
        prognostic = self._prog.increment_time(dt)
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    @abstractmethod
    def _update_prognostic_vars(self, prognostic: T) -> None: ...


TUVHT = TypeVar("TUVHT", bound=UVHT)


class BaseStateUVH(BaseState[TUVHT], Generic[TUVHT], metaclass=ABCMeta):
    """Base State for UVH models."""

    def __init__(self, prognostic: TUVHT) -> None:
        """Instantaite the state.

        Args:
            prognostic (TUVHT): Prognostic tuple.
        """
        super().__init__(prognostic)
        self._u = ZonalVelocity(prognostic.u)
        self._v = MeridionalVelocity(prognostic.v)
        self._h = LayerDepthAnomaly(prognostic.h)
        self._prog_vars[ZonalVelocity.get_name()] = self._u
        self._prog_vars[MeridionalVelocity.get_name()] = self._v
        self._prog_vars[LayerDepthAnomaly.get_name()] = self._h

    @property
    def u(self) -> ZonalVelocity:
        """Prognostic zonal velocity."""
        return self._u

    @property
    def v(self) -> MeridionalVelocity:
        """Prognostic meriodional velocity."""
        return self._v

    @property
    def h(self) -> LayerDepthAnomaly:
        """Prognostic layer thickness anomaly."""
        return self._h

    @abstractmethod
    def update_uvh(self, uvh: UVH) -> None:
        """Update u,v and h.

        Args:
            uvh (UVH): Prognostic u,v and h.
        """


TPSIQT = TypeVar("TPSIQT", bound=PSIQT)


class BaseStatePSIQ(BaseState[TPSIQT], Generic[TPSIQT], metaclass=ABCMeta):
    """Base State for PSIQ models."""

    def __init__(self, prognostic: TPSIQT) -> None:
        """Instantaite the state.

        Args:
            prognostic (TPSIQT): Prognostic tuple.
        """
        super().__init__(prognostic)
        self._psi = PrognosticStreamFunction(prognostic.psi)
        self._q = PrognosticPotentialVorticity(prognostic.q)
        self._prog_vars[PrognosticStreamFunction.get_name()] = self._psi
        self._prog_vars[PrognosticPotentialVorticity.get_name()] = self._q

    @property
    def psi(self) -> PrognosticStreamFunction:
        """Prognostic stream function."""
        return self._psi

    @property
    def q(self) -> PrognosticPotentialVorticity:
        """Prognostic potential vorticity."""
        return self._q

    @abstractmethod
    def update_psiq(self, psiq: PSIQ) -> None:
        """Update psi and q.

        Args:
            psiq (PSIQ): Prognostic psi and q.
        """


class StatePSIQ(BaseStatePSIQ[PSIQT]):
    """State: wrapper for PSIQT state variables.

    This wrapper links psiq variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def _update_prognostic_vars(self, prognostic: PSIQT) -> None:
        self._t.update(prognostic.t)
        self._psi.update(prognostic.psi)
        self._q.update(prognostic.q)

    def update_psiq(self, psiq: PSIQ) -> None:
        """Update psi and q.

        Args:
            psiq (PSIQ): Prognostic psi and q.
        """
        self.prognostic = PSIQT.from_psiq(self.t.get(), psiq)

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a steady state with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: StatePSIQT.
        """
        return cls(PSIQT.steady(n_ens, nl, nx, ny, dtype=dtype, device=device))

    @classmethod
    def from_tensors(
        cls,
        psi: torch.Tensor,
        q: torch.Tensor,
    ) -> Self:
        """Instantiate the state from tensors.

        Args:
            psi (torch.Tensor): Stream function.
            q (torch.Tensor): POtential vorticity.

        Returns:
            Self: StateUVH.
        """
        return cls(PSIQ(psi, q))


class StatePSIQAlpha(BaseStatePSIQ[PSIQTAlpha]):
    """State: wrapper for PSIQTAlpha state variables.

    This wrapper links psiq variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def __init__(self, prognostic: PSIQTAlpha) -> None:
        """Instantiate StatePSIQAlpha.

        Args:
            prognostic (PSIQTAlpha): Core prognostic variables.
        """
        super().__init__(prognostic)
        self._alpha = CollinearityCoefficient(prognostic.alpha)
        self._prog_vars[CollinearityCoefficient.get_name()] = self._alpha

    @property
    def alpha(self) -> CollinearityCoefficient:
        """Collinearity coefficient."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self.update_alpha(alpha)

    def _update_prognostic_vars(self, prognostic: PSIQTAlpha) -> None:
        """Update the prognostic variables.

        Args:
            prognostic (PSIQTAlpha): Prognostic tuple for psi and q.
        """
        self._t.update(prognostic.t)
        self._psi.update(prognostic.psi)
        self._q.update(prognostic.q)
        self._alpha.update(prognostic.alpha)

    def update_alpha(self, alpha: torch.Tensor) -> None:
        """Update only the value of alpha.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
        """
        for var in filter(lambda v: v.require_alpha, self.diag_vars.values()):
            var.outdated()
        prognostic = PSIQTAlpha.from_psiq(
            self.t.get(),
            alpha,
            self.prognostic.psiq,
        )
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    def update_psiq(self, psiq: PSIQ) -> None:
        """Update psi and q only.

        Args:
            psiq (PSIQ): Prognostic psi and q.
        """
        self.prognostic = PSIQTAlpha.from_psiq(
            self.t.get(),
            self.alpha.get(),
            psiq,
        )

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a steady state with zero-filled prognostic variables.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: StatePSIQTAlpha.
        """
        return cls(
            PSIQTAlpha.steady(n_ens, nl, nx, ny, dtype=dtype, device=device),
        )

    @classmethod
    def from_tensors(
        cls,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Self:
        """Instantiate the state from tensors.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meridional velocity.
            h (torch.Tensor): Surface height anomaly.
            alpha (torch.Tensor): Collinearity coefficient.

        Returns:
            Self: StatePSIQTAlpha.
        """
        return cls(PSIQTAlpha(u, v, h, alpha))


class StateUVH(BaseStateUVH[UVHT]):
    """State: wrapper for UVH state variables.

    This wrapper links uvh variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def _update_prognostic_vars(self, prognostic: UVHT) -> None:
        self._t.update(prognostic.t)
        self._u.update(prognostic.u)
        self._v.update(prognostic.v)
        self._h.update(prognostic.h)

    def update_uvh(self, uvh: UVH) -> None:
        """Update u,v and h.

        Args:
            uvh (UVH): Prognostic u,v and h.
        """
        self.prognostic = UVHT.from_uvh(self.t.get(), uvh)

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a steady state with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: StateUVH.
        """
        return cls(UVHT.steady(n_ens, nl, nx, ny, dtype=dtype, device=device))

    @classmethod
    def from_tensors(
        cls,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> Self:
        """Instantiate the state from tensors.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meridional velocity.
            h (torch.Tensor): Surface height anomaly.

        Returns:
            Self: StateUVH.
        """
        return cls(UVH(u, v, h))


class StateUVHAlpha(BaseStateUVH[UVHTAlpha]):
    """StateUVHAlpha: wrapper for UVHTAlpha state variables.

    This wrapper links uvh variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def __init__(self, prognostic: UVHTAlpha) -> None:
        """Instantiate StateUVHAlpha.

        Args:
            prognostic (UVHTAlpha): Core prognostic variables.
        """
        super().__init__(prognostic)
        self._alpha = CollinearityCoefficient(prognostic.alpha)
        self._prog_vars[CollinearityCoefficient.get_name()] = self._alpha

    @property
    def alpha(self) -> CollinearityCoefficient:
        """Collinearity coefficient."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self.update_alpha(alpha)

    def _update_prognostic_vars(self, prognostic: UVHTAlpha) -> None:
        self._t.update(prognostic.t)
        self._u.update(prognostic.u)
        self._v.update(prognostic.v)
        self._h.update(prognostic.h)
        self._alpha.update(prognostic.alpha)

    def update_alpha(self, alpha: torch.Tensor) -> None:
        """Update only the value of alpha.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
        """
        for var in filter(lambda v: v.require_alpha, self.diag_vars.values()):
            var.outdated()
        prognostic = UVHTAlpha.from_uvh(
            self.t.get(),
            alpha,
            self.prognostic.uvh,
        )
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    def update_uvh(self, uvh: UVH) -> None:
        """Update u,v and h only.

        Args:
            uvh (UVH): Prognostic u,v and h.
        """
        self.prognostic = UVHTAlpha.from_uvh(
            self.t.get(),
            self.alpha.get(),
            uvh,
        )

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> Self:
        """Instantiate a steady state with zero-filled prognostic variables.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype, optional): Data type. Defaults to None.
            device (torch.device, optional): Device to use. Defaults to None.

        Returns:
            Self: StateUVH.
        """
        return cls(
            UVHTAlpha.steady(n_ens, nl, nx, ny, dtype=dtype, device=device),
        )

    @classmethod
    def from_tensors(
        cls,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Self:
        """Instantiate the state from tensors.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meridional velocity.
            h (torch.Tensor): Surface height anomaly.
            alpha (torch.Tensor): Collinearity coefficient.

        Returns:
            Self: StateUVH.
        """
        return cls(UVHTAlpha(u, v, h, alpha))
