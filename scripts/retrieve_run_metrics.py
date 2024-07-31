"""Retrieve a Run Consumption."""

# ruff: noqa: ERA001

import json
import os
from datetime import datetime
from pathlib import Path

import requests
import toml
from dateutil import tz
from dotenv import load_dotenv


def parse_json(power_json: dict) -> dict:
    """Parse results Json.

    Args:
        power_json (dict): Power Json.

    Returns:
        dict: Parsed Json.
    """
    for data in power_json:
        time = data["timestamp"]
        data["timestamp"] = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f%z")
    return power_json


def get_metric(
    node: str,
    site: str,
    start: int,
    stop: int,
    metric: str,
) -> dict:
    """Retrieve Power consumption from Grid5000 jobs.

    Args:
        node (str): Node name.
        site (str): Site.
        start (int): Starting time (in seconds)
        stop (int): Stopping time (in seconds).
        metric (str, optional): Metric to monitor.

    Returns:
        dict: Results.
    """
    url = f"https://api.grid5000.fr/stable/sites/{site}/metrics?metrics={metric}&nodes={node}&start_time={int(start)}&end_time={int(stop)}"
    data = session.get(url, verify=True)  # .json()
    data.raise_for_status()
    return data.json()


def read_power(file: Path) -> dict:
    """Read power consumption from a file.

    Args:
        file (Path): _description_

    Returns:
        dict: _description_
    """
    return parse_json(json.load(file.open("r")))


tz_fr = tz.gettz("Europe/France")

if __name__ == "__main__":
    load_dotenv()

    session = requests.Session()
    session.auth = (os.environ["G5K_LOGIN"], os.environ["G5K_PASSWORD"])

    ROOT = Path(__file__).parent.parent

    with ROOT.joinpath("config/retrieve_run_metrics.toml").open("r") as file:
        config = toml.load(file)

    folder = ROOT.joinpath(f"{config['folder']}")
    for metric in config["metrics"]:
        print(f"Collecting: {metric}")  # noqa: T201
        for run in config["runs"]:
            print(f"\tRetrieving for job n°{run['id']}")  # noqa: T201
            results = get_metric(
                "abacus16-1",
                "rennes",
                run["start"].timestamp(),
                run["end"].timestamp(),
                metric=metric,
            )
            output = folder.joinpath(
                f"{run['folder']}/{metric}_{run['id']}.json",
            )

            if not output.parent.is_dir():
                output.parent.mkdir(parents=True)

            with output.open("w") as file:
                json.dump(results, file)
