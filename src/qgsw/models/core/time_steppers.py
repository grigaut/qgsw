"""Schemes for Time-integration."""

from collections.abc import Callable
from typing import TypeVar, Union

from qgsw.fields.variables.tuples import PSIQ, UVH

Prognostic = TypeVar("Prognostic", bound=Union[UVH, PSIQ])

AVAILABLE = ["rk3", "euler"]


def rk3_ssp(
    prog: Prognostic,
    dt: float,
    time_derivation_func: Callable[[Prognostic], Prognostic],
) -> Prognostic:
    """Perform time-integration using a RK3-SSP scheme..

    Args:
        prog (Prognostic): Prognostic variables.
        dt (float): Timestep.
        time_derivation_func (Callable): Time derivation.

    Returns:
        Prognostic: Final prog variables.
    """
    dt0_prog = time_derivation_func(prog)
    prog += dt * dt0_prog

    dt1_prog = time_derivation_func(prog)
    prog += (dt / 4) * (dt1_prog - 3 * dt0_prog)

    dt2_prog = time_derivation_func(prog)
    prog += (dt / 12) * (8 * dt2_prog - dt1_prog - dt0_prog)

    return prog


def euler(
    prog: Prognostic,
    dt: float,
    time_derivation_func: Callable[[Prognostic], Prognostic],
) -> Prognostic:
    """Perform time-integration using a Euler scheme..

    Args:
        prog (Prognostic): Prognostic variables.
        dt (float): Timestep.
        time_derivation_func (Callable): Time derivation.

    Returns:
        Prognostic: Final prog variables.
    """
    dt_prog = time_derivation_func(prog)
    prog += dt * dt_prog
    return prog


def validate(time_stepper: str) -> None:
    """Validate time stepper.

    Args:
        time_stepper (str): Time stepper name.

    Raises:
        ValueError: If the time stepper is not available.
    """
    if time_stepper not in AVAILABLE:
        msg = f"Invalid time stepper: {time_stepper}"
        raise ValueError(msg)
