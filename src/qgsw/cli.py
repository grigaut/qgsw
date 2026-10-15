"""Command Line Interface."""

from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path
from typing import Any

from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.vardyn.core import MIOSTBasis
from qgsw.decomposition.wavelets.core import WaveletBasis
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


def parse_presets() -> dict[str, Any]:
    """Parse presets."""
    parser = argparse.ArgumentParser(
        description="Retrieve preset.",
    )
    parser.add_argument("--presets", type=pathlib.Path, help="Presets file")
    args, _ = parser.parse_known_args()
    if (presets := args.presets) is None:
        return {}
    with presets.open("r") as f:
        return json.load(f)


class ScriptsArgsParser:
    """Scripts arguments parser."""

    retrieved = False

    has_config = False
    has_verbose = False
    has_indices = False
    has_comparison = False
    has_cycles = False
    has_prefix = False
    has_no_wind = False
    has_obs_track = False
    has_optim = False
    has_separation = False
    has_season = False
    has_gamma_reg_dyn = False
    has_gamma_reg_bkg = False
    has_no_reg = False
    has_no_alpha = False
    has_reg_exp = False
    has_wind_optim = False
    has_gamma_sst = False
    has_no_ml_optim = False
    has_sst_forcing = False
    has_gamma_sst_reg = False
    has_dt = False
    has_dx = False
    has_dy = False
    has_sigma_ic = False
    has_sigma_bc = False
    has_basis_dyn = False
    has_basis_ext = False
    has_model_type = False

    @property
    def config(self) -> Path:
        """Configuration."""
        self._check_attr(self.has_config, "Configuration not tracked.")
        return self.namespace.config

    @property
    def verbose(self) -> int:
        """Verbose level."""
        self._check_attr(self.has_verbose, "Verbose not tracked.")
        return self.namespace.verbose

    @property
    def indices(self) -> tuple[int, int, int, int]:
        """Indices."""
        self._check_attr(self.has_indices, "Indices not tracked.")
        return self.namespace.indices

    @property
    def comparison(self) -> int:
        """Comparison interval."""
        self._check_attr(
            self.has_comparison,
            "Comparison interval not tracked.",
        )
        return self.namespace.comparison

    @property
    def cycles(self) -> int:
        """Number of cycles."""
        self._check_attr(
            self.has_cycles,
            "Cycles not tracked.",
        )
        return self.namespace.cycles

    @property
    def prefix(self) -> str:
        """Prefix."""
        self._check_attr(
            self.has_prefix,
            "Prefix not tracked.",
        )
        return self.namespace.prefix

    @property
    def no_wind(self) -> bool:
        """No wind."""
        self._check_attr(
            self.has_no_wind,
            "Wind not tracked.",
        )
        return self.namespace.no_wind

    @property
    def obs_track(self) -> bool:
        """Observation tracks."""
        self._check_attr(
            self.has_obs_track,
            "Observation tracks not tracked.",
        )
        return self.namespace.obs_track

    @property
    def optim(self) -> int:
        """Max optimization steps."""
        self._check_attr(
            self.has_optim,
            "Max optimization steps tracks not tracked.",
        )
        return self.namespace.optim

    @property
    def separation(self) -> int:
        """Separation steps."""
        self._check_attr(
            self.has_separation,
            "Separation steps not tracked.",
        )
        return self.namespace.separation

    @property
    def season(self) -> str:
        """Seasons."""
        self._check_attr(
            self.has_season,
            "Seasons not tracked.",
        )
        return self.namespace.season

    @property
    def gamma_reg_dyn(self) -> float:
        """Gamma_reg_dyn."""
        self._check_attr(
            self.has_gamma_reg_dyn,
            "Gamma_reg_dyn not tracked.",
        )
        return self.namespace.gamma_reg_dyn

    @property
    def gamma_reg_bkg(self) -> float:
        """Gamma_reg_bkg."""
        self._check_attr(
            self.has_gamma_reg_bkg,
            "Gamma_reg_bkg not tracked.",
        )
        return self.namespace.gamma_reg_bkg

    @property
    def no_reg(self) -> bool:
        """No regularization."""
        self._check_attr(
            self.has_no_reg,
            "Regularization not tracked.",
        )
        return self.namespace.no_reg

    @property
    def no_alpha(self) -> bool:
        """No alpha."""
        self._check_attr(
            self.has_no_alpha,
            "Alpha not tracked.",
        )
        return self.namespace.no_alpha

    @property
    def reg_exp(self) -> int:
        """reg_exp."""
        self._check_attr(
            self.has_reg_exp,
            "reg_exp not tracked.",
        )
        return self.namespace.reg_exp

    @property
    def wind_optim(self) -> bool:
        """wind_optim."""
        self._check_attr(
            self.has_wind_optim,
            "wind_optim not tracked.",
        )
        return self.namespace.wind_optim

    @property
    def gamma_sst(self) -> float:
        """gamma_sst."""
        self._check_attr(
            self.has_gamma_sst,
            "gamma_sst not tracked.",
        )
        return self.namespace.gamma_sst

    @property
    def no_ml_optim(self) -> bool:
        """no_ml_optim."""
        self._check_attr(
            self.has_no_ml_optim,
            "no_ml_optim not tracked.",
        )
        return self.namespace.no_ml_optim

    @property
    def with_sst_forcing(self) -> bool:
        """with_sst_forcing."""
        self._check_attr(
            self.has_sst_forcing,
            "with_sst_forcing not tracked.",
        )
        return self.namespace.with_sst_forcing

    @property
    def gamma_sst_reg(self) -> float:
        """gamma_sst_reg."""
        self._check_attr(
            self.has_gamma_sst_reg,
            "gamma_sst_reg not tracked.",
        )
        return self.namespace.gamma_sst_reg

    @property
    def dt(self) -> int:
        """Dt."""
        self._check_attr(
            self.has_dt,
            "dt not tracked.",
        )
        return self.namespace.dt

    @property
    def dx(self) -> int:
        """Dx."""
        self._check_attr(
            self.has_dx,
            "dx not tracked.",
        )
        return self.namespace.dx

    @property
    def dy(self) -> int:
        """Dy."""
        self._check_attr(
            self.has_dy,
            "dy not tracked.",
        )
        return self.namespace.dy

    @property
    def sigma_ic(self) -> float:
        """sigma_ic."""
        self._check_attr(
            self.has_sigma_ic,
            "sigma_ic not tracked.",
        )
        return self.namespace.sigma_ic

    @property
    def sigma_bc(self) -> float:
        """sigma_bc."""
        self._check_attr(
            self.has_sigma_bc,
            "sigma_bc not tracked.",
        )
        return self.namespace.sigma_bc

    @property
    def basis_dyn(self) -> str:
        """Basis dynamical."""
        self._check_attr(
            self.has_basis_dyn,
            "basis_dyn not tracked.",
        )
        return self.namespace.basis_dyn

    @property
    def basis_ext(self) -> str:
        """Basis external."""
        self._check_attr(
            self.has_basis_ext,
            "basis_ext not tracked.",
        )
        return self.namespace.basis_ext

    @property
    def model_type(self) -> str:
        """Model type."""
        self._check_attr(
            self.has_model_type,
            "model_type not tracked.",
        )
        return self.namespace.model_type

    def __init__(self) -> None:
        """Instantiate parser."""
        self.parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        self.namespace = argparse.Namespace()

    def _check_unretrieved(self) -> None:
        if self.retrieved:
            msg = "Arguments already retrieved yet."
            raise ValueError(msg)

    def _check_attr(self, has_attr: bool, msg: str) -> None:  # noqa: FBT001
        if not self.retrieved:
            msg = "Argument not retrieved yet, use ScriptArgs.retrieve()."
            raise ValueError(msg)
        if has_attr:
            return
        raise AttributeError(msg)

    def add_config(self) -> None:
        """Add configuration to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--config",
            type=pathlib.Path,
            help="Configuration File Path (from qgsw root level)",
        )
        self.has_config = True

    def add_verbose(self) -> None:
        """Add verbose to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Verbose level.",
        )
        self.has_verbose = True

    def add_indices(self) -> None:
        """Add indices to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--indices",
            "-i",
            required=False,
            nargs=4,
            type=int,
            help="Indices (imin, imax, jmin, jmax), "
            "for example (64, 128, 128, 256).",
        )
        self.has_indices = True

    def add_comparison_interval(self, default: int = 1) -> None:
        """Add comparison interval to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int, optional): Default value for for the comparison
                interval. Defaults to 1.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "-c",
            "--comparison",
            type=int,
            default=default,
            help="Comparison interval value.",
        )
        self.has_comparison = True

    def add_cycles(self, default: int = 3) -> None:
        """Add number of cycles to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value for for the comparison interval.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--cycles",
            type=int,
            default=default,
            help="Number of cycles.",
        )
        self.has_cycles = True

    def add_prefix(self, default: str = "results") -> None:
        """Add output file prefix to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value for for the prefix.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "-p",
            "--prefix",
            type=str,
            default=default,
            help="File saving prefix prefix.",
        )
        self.has_prefix = True

    def add_wind(self) -> None:
        """Specify whether to use wind or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--no-wind",
            action="store_true",
            help="Disable wind forcing.",
        )
        self.has_no_wind = True

    def add_obs_track(self) -> None:
        """Specify whether to use observation trackes or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--obs-track",
            action="store_true",
            help="Enable observations tracks.",
        )
        self.has_obs_track = True

    def add_optim_max_step(self, default: int = 200) -> None:
        """Specify number of optimization steps to perform.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--optim",
            "-o",
            type=int,
            default=default,
            help="Max optimization steps.",
        )
        self.has_optim = True

    def add_separation(self, default: int = 0) -> None:
        """Specify number of step to perform to separate cycles.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int, optional): Default value. Defaults to 0.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--separation",
            "-s",
            type=int,
            default=default,
            help="Number of step to separate cycles.",
        )
        self.has_separation = True

    def add_season(self, default: str = "summer") -> None:
        """Specify season.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (str | None): Default season. Defaults to "summer".
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--season",
            type=str,
            default=default,
            choices=["summer", "autumn", "winter", "spring"],
            help="Season.",
        )
        self.has_season = True

    def add_gamma_reg_dyn(self, default: float = 1) -> None:
        """Dynamical regularization coef.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (float, optional): Default value. Defaults to 1.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--gamma-reg-dyn",
            "--gamma-dyn",
            "--grdyn",
            type=float,
            default=default,
            help="Gamma reg dyn value.",
        )
        self.has_gamma_reg_dyn = True

    def add_gamma_reg_bkg(self, default: float = 1) -> None:
        """Background regularization coef.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (float, optional): Default value. Defaults to 1.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--gamma-reg-bkg",
            "--gamma-bkg",
            "--grbkg",
            type=float,
            default=default,
            help="Gamma reg bkg value.",
        )
        self.has_gamma_reg_bkg = True

    def add_no_reg(self) -> None:
        """Specify whether to use regularization or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--no-reg",
            action="store_true",
            help="Disable regularization.",
        )
        self.has_no_reg = True

    def add_regularization(
        self,
        gamma_dyn_default: float = 1,
        gamma_bkg_default: float = 1,
    ) -> None:
        """Add regularization."""
        self.add_gamma_reg_dyn(gamma_dyn_default)
        self.add_gamma_reg_bkg(gamma_bkg_default)
        self.add_no_reg()

    def add_alpha(self) -> None:
        """Specify whether to use alpha or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--no-alpha",
            action="store_true",
            help="Disable regularization.",
        )
        self.has_no_alpha = True

    def add_reg_exp(self, default: int = 2) -> None:
        """Add regularization exponent."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--reg-exp",
            type=int,
            default=default,
            help="Regularization exponent value.",
        )
        self.has_reg_exp = True

    def add_wind_optim(self) -> None:
        """Add wind optimization."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--wind-optim",
            action="store_true",
            help="Whether to perform wind optimization.",
        )
        self.has_wind_optim = True

    def add_gamma_sst(self, default: float = 1) -> None:
        """Add gamma sst."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--gamma-sst",
            type=float,
            default=default,
            help="Gamma for SST loss.",
        )
        self.has_gamma_sst = True

    def add_no_ml_optim(self) -> None:
        """Add ML optimization."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--no-ml-optim",
            action="store_true",
            help="Whether to perform ML optimization.",
        )
        self.has_no_ml_optim = True

    def add_with_sst_forcing(self) -> None:
        """Add SST forcing option."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--with-sst-forcing",
            action="store_true",
            help="Whether to use SST forcing.",
        )
        self.has_sst_forcing = True

    def add_gamma_sst_reg(self, default: float = 1) -> None:
        """Add gamma sst reg."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--gamma-sst-reg",
            type=float,
            default=default,
            help="Gamma for SST regularization loss.",
        )
        self.has_gamma_sst_reg = True

    def add_dt(self, default: int) -> None:
        """Add dt."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--dt",
            type=int,
            default=default,
            help="Timestep.",
        )
        self.has_dt = True

    def add_dx(self, default: int) -> None:
        """Add dx."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--dx",
            type=int,
            default=default,
            help="X step.",
        )
        self.has_dx = True

    def add_dy(self, default: int) -> None:
        """Add dy."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--dy",
            type=int,
            default=default,
            help="Y step.",
        )
        self.has_dy = True

    def add_sigma_ic(self, default: float) -> None:
        """Add sigma_ic."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--sigma-ic",
            "--ic",
            type=float,
            default=default,
            help="Initial condition filtering std.",
        )
        self.has_sigma_ic = True

    def add_sigma_bc(self, default: float) -> None:
        """Add sigma_bc."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--sigma-bc",
            "--bc",
            type=float,
            default=default,
            help="Boundary condition filtering std.",
        )
        self.has_sigma_bc = True

    def add_basis_dynamical(self) -> None:
        """Add basis_dyn to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--basis-dyn",
            "--bdyn",
            required=False,
            type=str,
            choices=[
                MIOSTBasis.type,
                WaveletBasis.type,
                GaussianExpBasis.type,
            ],
            help="Basis dynamical identification.",
        )
        self.has_basis_dyn = True

    def add_basis_external(self) -> None:
        """Add basis_ext to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--basis-ext",
            "--bext",
            required=False,
            type=str,
            choices=[
                MIOSTBasis.type,
                WaveletBasis.type,
                GaussianExpBasis.type,
            ],
            help="Basis external identification.",
        )
        self.has_basis_ext = True

    def add_model_type(self) -> None:
        """Add model-type to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "-m",
            "--model-type",
            type=str,
            choices=[
                QGPSIQForced.get_type(),
                QGPSIQRGPsi2TransportDR.get_type(),
                QGPSIQForcedSurfML.get_type(),
            ],
            help="Model type.",
        )
        self.has_model_type = True

    def retrieve(self) -> None:
        """Retrieve arguments."""
        self.parser.parse_known_args(namespace=self.namespace)
        self.retrieved = True

    def set_presets(self, presets: dict | None = None) -> None:
        """Set presets."""
        if not presets:
            return
        self.parser.set_defaults(**presets)

    @classmethod
    def va_setup(
        cls,
        prefix_default: str,
        cycles_default: int = 3,
    ) -> Self:
        """Pre-setup parser for variationla assimilaiton scripts."""
        obj = cls()
        obj.add_config()
        obj.add_verbose()
        obj.add_prefix(default=prefix_default)
        obj.add_comparison_interval(default=1)
        obj.add_wind()
        obj.add_obs_track()
        obj.add_cycles(default=cycles_default)
        obj.add_optim_max_step(default=200)
        obj.add_separation(default=0)
        obj.add_model_type()
        obj.add_basis_dynamical()
        obj.add_basis_external()
        return obj
