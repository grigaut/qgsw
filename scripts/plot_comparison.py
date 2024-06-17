"""Plot Models Comparison."""

from pathlib import Path

import numpy as np
import toml
from icecream import ic
from qgsw.run_summary import RunSummary

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT.joinpath("config/plot_comparison.toml")

with CONFIG_FILE.open() as f:
    config = toml.load(f)

summaries: list[RunSummary] = []
for folder in config["folders"]:
    summary_path = ROOT.joinpath(folder + "_summary.toml")
    summaries.append(RunSummary.from_file(summary_path))

if len(np.unique([summary.dt for summary in summaries])) != 1:
    ic("Error")

if len(np.unique([summary.duration for summary in summaries])) != 1:
    ic("Error")
