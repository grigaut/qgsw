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
    has_no_reg = False
    has_no_alpha = False
    has_alpha_lr = False
    has_wind_optim = False
    has_wind_lr = False
    has_gamma_sst = False
    has_no_ml_optim = False
    has_sst_forcing = False
    has_gamma_sst_reg = False
    has_dt = False
    has_dx = False
    has_dy = False
    has_sigma_ic = False
    has_sigma_bc = False
    has_model_type = False
    has_perturbed_params = False
    has_dyn_reg_qg = False
    has_dyn_reg_div = False
    has_gamma_dyn_qg = False
    has_gamma_dyn_div = False
    has_dry_run = False

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
    def alpha_lr(self) -> float:
        """alpha_lr."""
        self._check_attr(
            self.has_alpha_lr,
            "Alpha_lr not tracked.",
        )
        return self.namespace.alpha_lr

    @property
    def wind_optim(self) -> bool:
        """wind_optim."""
        self._check_attr(
            self.has_wind_optim,
            "wind_optim not tracked.",
        )
        return self.namespace.wind_optim

    @property
    def wind_lr(self) -> float:
        """wind_lr."""
        self._check_attr(
            self.has_wind_lr,
            "wind_lr not tracked.",
        )
        return self.namespace.wind_lr

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
    def model_type(self) -> str:
        """Model type."""
        self._check_attr(
            self.has_model_type,
            "model_type not tracked.",
        )
        return self.namespace.model_type

    @property
    def perturbed_params(self) -> bool:
        """Perturbed params."""
        self._check_attr(
            self.has_perturbed_params,
            "perturbed_params not tracked.",
        )
        return self.namespace.perturbed_params

    @property
    def dyn_reg_qg(self) -> bool:
        """QG dynamical regularization."""
        self._check_attr(
            self.has_dyn_reg_qg,
            "dyn_reg_qg not tracked.",
        )
        return self.namespace.dyn_reg_qg

    @property
    def dyn_reg_div(self) -> bool:
        """Divergence dynamical regularization."""
        self._check_attr(
            self.has_dyn_reg_div,
            "dyn_reg_div not tracked.",
        )
        return self.namespace.dyn_reg_div

    @property
    def gamma_dyn_qg(self) -> bool:
        """Divergence dynamical regularization coef."""
        self._check_attr(
            self.has_gamma_dyn_qg,
            "gamma_dyn_qg not tracked.",
        )
        return self.namespace.gamma_dyn_qg

    @property
    def gamma_dyn_div(self) -> bool:
        """Divergence dynamical regularization coef."""
        self._check_attr(
            self.has_gamma_dyn_div,
            "gamma_dyn_div not tracked.",
        )
        return self.namespace.gamma_dyn_div

    @property
    def dry_run(self) -> bool:
        """Dry run."""
        self._check_attr(
            self.has_dry_run,
            "dry_run not tracked.",
        )
        return self.namespace.dry_run

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

    def add_alpha(self, lr_default: float) -> None:
        """Specify whether to use alpha or not."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--no-alpha",
            action="store_true",
            help="Disable regularization.",
        )
        self.has_no_alpha = True
        self.add_alpha_lr(lr_default)

    def add_alpha_lr(self, default_value: float) -> None:
        """Alpha learning rate."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--alpha-lr",
            type=float,
            default=default_value,
            help="Learning rate for alpha.",
        )
        self.has_alpha_lr = True

    def add_wind_optim(self, lr_default: float) -> None:
        """Add wind optimization."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--wind-optim",
            action="store_true",
            help="Whether to perform wind optimization.",
        )
        self.has_wind_optim = True
        self.add_wind_lr(lr_default)

    def add_wind_lr(self, default_value: float) -> None:
        """Wind learning rate."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--wind-lr",
            type=float,
            default=default_value,
            help="Learning rate for wind.",
        )
        self.has_wind_lr = True

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

    def add_dtdxdy(
        self, dt_default: float, dx_default: int, dy_default: int
    ) -> None:
        """Add dt, dx and dy."""
        self.add_dt(dt_default)
        self.add_dx(dx_default)
        self.add_dy(dy_default)

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

    def add_sigma_icbc(self, ic_default: float, bc_default: float) -> None:
        """Add sigma_ic and sigma_bc."""
        self.add_sigma_ic(ic_default)
        self.add_sigma_bc(bc_default)

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

    def add_perturbed_params(self) -> None:
        """Add perturbed_params."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--perturbed-params",
            action="store_true",
            help="Whether to perturb params.",
        )
        self.has_perturbed_params = True

    def add_dyn_reg_qg(self) -> None:
        """Dynamical regularization QG."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--dyn-reg-qg",
            action="store_true",
            help="Whether to use QG dynmical regularization.",
        )
        self.has_dyn_reg_qg = True

    def add_gamma_reg_dyn_qg(self, default: float = 1) -> None:
        """Dynamical regularization coef.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (float, optional): Default value. Defaults to 1.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--gamma-dyn-qg",
            type=float,
            default=default,
            help="Gamma reg dyn qg value.",
        )
        self.has_gamma_dyn_qg = True

    def add_dyn_reg_div(self) -> None:
        """Dynamical regularization div."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--dyn-reg-div",
            action="store_true",
            help="Whether to use div dynmical regularization.",
        )
        self.has_dyn_reg_div = True

    def add_gamma_reg_dyn_div(self, default: float = 1) -> None:
        """Dynamical regularization coef.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (float, optional): Default value. Defaults to 1.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--gamma-dyn-div",
            type=float,
            default=default,
            help="Gamma reg dyn div value.",
        )
        self.has_gamma_dyn_div = True

    def add_regs(
        self, dyn_qg_default: float = 1, dyn_div_default: float = 1
    ) -> None:
        """Add regularizations."""
        self.add_no_reg()
        self.add_dyn_reg_qg()
        self.add_gamma_reg_dyn_qg(dyn_qg_default)
        self.add_dyn_reg_div()
        self.add_gamma_reg_dyn_div(dyn_div_default)

    def add_dry_run(self) -> None:
        """Dry run."""
        self._check_unretrieved()
        self.parser.add_argument(
            "--dry-run",
            action="store_true",
            help="To perform dry run.",
        )
        self.has_dry_run = True

    def retrieve(self) -> None:
        """Retrieve arguments."""
        self.parser.parse_known_args(namespace=self.namespace)
        self.retrieved = True

    def set_presets(self, presets: dict | None = None) -> None:
        """Set presets."""
        if not presets:
            return
        tracked_dests = {
            action.dest
            for action in self.parser._actions  # noqa: SLF001
            if action.dest
        }
        filtered_presets = {
            k: v for k, v in presets.items() if k in tracked_dests
        }
        self.parser.set_defaults(**filtered_presets)

    @classmethod
    def va_setup(
        cls,
        prefix_default: str,
        cycles_default: int = 3,
    ) -> Self:
        """Pre-setup parser for variationla assimilaiton scripts."""
        obj = cls()
        obj.add_dry_run()
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
        return obj


class BasisParser:
    """Basis parser."""

    retrieved = False

    @property
    def type(self) -> str:
        """Basis type."""
        return getattr(self.namespace, f"basis_{self.basis_name}")

    @property
    def learning_rate(self) -> float:
        """Learning rate."""
        return getattr(self.namespace, f"basis_{self.basis_name}_lr")

    @property
    def with_background(self) -> bool:
        """With background regularization."""
        return getattr(self.namespace, f"basis_{self.basis_name}_bkg_reg")

    @property
    def gamma(self) -> float:
        """Gamma."""
        return getattr(self.namespace, f"basis_{self.basis_name}_gamma")

    @property
    def optim(self) -> bool:
        """Whether to optim the basis coefficients or not."""
        return not getattr(self.namespace, f"basis_{self.basis_name}_no_optim")

    def __repr__(self) -> str:
        """String representation."""
        return self.namespace.__repr__()

    def __init__(self, basis_name: str) -> None:
        """Instantiate parser."""
        self.basis_name = basis_name
        self.parser = argparse.ArgumentParser(
            description="Retrieve basis arguments.",
        )
        self.namespace = argparse.Namespace()
        ## Basis type
        self.parser.add_argument(
            f"--basis-{basis_name}",
            f"--b{basis_name}",
            type=str,
            choices=[
                MIOSTBasis.type,
                WaveletBasis.type,
                GaussianExpBasis.type,
            ],
            help=f"Basis '{basis_name}' identification.",
        )
        ## Basis learning rate
        self.parser.add_argument(
            f"--basis-{basis_name}-lr",
            f"--b{basis_name}-lr",
            type=float,
            help=f"Learning rate for b{basis_name}.",
        )
        ### Basis background regularization
        self.parser.add_argument(
            f"--basis-{basis_name}-bkg-reg",
            f"--b{basis_name}-bkg-reg",
            action="store_true",
            help=f"Whether to use background regularization for b{basis_name}.",  # noqa: E501
        )
        ### Basis gamma for background
        self.parser.add_argument(
            f"--basis-{basis_name}-gamma",
            f"--b{basis_name}-gamma",
            type=float,
            help=f"Background gamma for b{basis_name}.",
        )
        ### Basis optim
        self.parser.add_argument(
            f"--basis-{basis_name}-no-optim",
            action="store_true",
            help=f"Whether to optim b{basis_name} coefficients.",
        )

    def retrieve(self) -> None:
        """Retrieve arguments."""
        self.parser.parse_known_args(namespace=self.namespace)

    def set_presets(self, presets: dict | None = None) -> None:
        """Set presets."""
        if not presets:
            return
        tracked_dests = {
            action.dest
            for action in self.parser._actions  # noqa: SLF001
            if action.dest
        }
        filtered_presets = {
            k: v for k, v in presets.items() if k in tracked_dests
        }
        self.parser.set_defaults(**filtered_presets)
