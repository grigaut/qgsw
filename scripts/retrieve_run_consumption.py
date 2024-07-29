"""Retrieve a Run Consumption."""

# ruff: noqa: ERA001

import json
import os
from datetime import datetime
from pathlib import Path

import requests
from dateutil import tz
from dotenv import load_dotenv

load_dotenv()

session = requests.Session()
session.auth = (os.environ["G5K_LOGIN"], os.environ["G5K_PASSWORD"])


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


def get_power(
    node: str,
    site: str,
    start: int,
    stop: int,
    metric: str = "bmc_node_power_watt",
) -> dict:
    """Retrieve Power consumption from Grid5000 jobs.

    Args:
        node (str): Node name.
        site (str): Site.
        start (int): Starting time (in seconds)
        stop (int): Stopping time (in seconds).
        metric (str, optional): Metric to monitor.
        Defaults to "bmc_node_power_watt".

    Returns:
        dict: Results.
    """
    url = f"https://api.grid5000.fr/stable/sites/{site}/metrics?metrics={metric}&nodes={node}&start_time={int(start)}&end_time={int(stop)}"
    data = session.get(url, verify=False)  # .json()
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

parameters = [
    {
        "nb": "2263719",
        "start": datetime(2024, 7, 26, 17, 32, 23, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 17, 48, 35, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263720",
        "start": datetime(2024, 7, 26, 17, 48, 57, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 18, 5, 5, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263721",
        "start": datetime(2024, 7, 26, 18, 5, 22, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 18, 21, 31, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263722",
        "start": datetime(2024, 7, 26, 18, 21, 47, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 18, 38, 4, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263723",
        "start": datetime(2024, 7, 26, 18, 38, 19, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 18, 54, 29, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263724",
        "start": datetime(2024, 7, 26, 18, 54, 53, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 19, 10, 42, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263725",
        "start": datetime(2024, 7, 26, 19, 11, 5, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 19, 27, 0, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263726",
        "start": datetime(2024, 7, 26, 19, 27, 15, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 19, 43, 27, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263727",
        "start": datetime(2024, 7, 26, 19, 43, 42, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 19, 59, 40, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263728",
        "start": datetime(2024, 7, 26, 19, 59, 55, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 20, 16, 5, tzinfo=tz_fr),
        "folder": "one_layer_model",
    },
    {
        "nb": "2263734",
        "start": datetime(2024, 7, 26, 20, 16, 23, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 20, 33, 19, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263735",
        "start": datetime(2024, 7, 26, 20, 33, 42, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 20, 50, 29, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263736",
        "start": datetime(2024, 7, 26, 20, 50, 52, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 21, 7, 56, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263736",
        "start": datetime(2024, 7, 26, 21, 8, 11, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 21, 25, 1, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263738",
        "start": datetime(2024, 7, 26, 21, 25, 16, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 21, 42, 15, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263739",
        "start": datetime(2024, 7, 26, 21, 42, 38, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 21, 59, 24, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263740",
        "start": datetime(2024, 7, 26, 21, 59, 39, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 22, 16, 41, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263741",
        "start": datetime(2024, 7, 26, 22, 16, 55, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 22, 33, 48, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263742",
        "start": datetime(2024, 7, 26, 22, 34, 7, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 22, 50, 52, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263743",
        "start": datetime(2024, 7, 26, 22, 51, 14, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 23, 7, 56, tzinfo=tz_fr),
        "folder": "two_layers_model",
    },
    {
        "nb": "2263751",
        "start": datetime(2024, 7, 26, 23, 8, 12, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 23, 24, 31, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263752",
        "start": datetime(2024, 7, 26, 23, 24, 53, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 23, 41, 7, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263753",
        "start": datetime(2024, 7, 26, 23, 41, 22, tzinfo=tz_fr),
        "end": datetime(2024, 7, 26, 23, 57, 40, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263754",
        "start": datetime(2024, 7, 26, 23, 58, 3, tzinfo=tz_fr),
        "end": datetime(2024, 7, 27, 0, 14, 7, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263755",
        "start": datetime(2024, 7, 27, 0, 14, 26, tzinfo=tz_fr),
        "end": datetime(2024, 7, 27, 0, 30, 57, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263756",
        "start": datetime(2024, 7, 27, 0, 31, 12, tzinfo=tz_fr),
        "end": datetime(2024, 7, 27, 0, 47, 20, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263757",
        "start": datetime(2024, 7, 27, 0, 47, 35, tzinfo=tz_fr),
        "end": datetime(2024, 7, 27, 1, 9, 37, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263758",
        "start": datetime(2024, 7, 27, 1, 9, 56, tzinfo=tz_fr),
        "end": datetime(2024, 7, 27, 1, 26, 1, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263759",
        "start": datetime(2024, 7, 27, 1, 26, 25, tzinfo=tz_fr),
        "end": datetime(2024, 7, 27, 1, 42, 59, tzinfo=tz_fr),
        "folder": "modified_model",
    },
    {
        "nb": "2263760",
        "start": datetime(2024, 7, 27, 1, 43, 17, tzinfo=tz_fr),
        "end": datetime(2024, 7, 27, 1, 59, 39, tzinfo=tz_fr),
        "folder": "modified_model",
    },
]

for run in parameters:
    results = get_power(
        "abacus16-1",
        "rennes",
        run["start"].timestamp(),
        run["end"].timestamp(),
    )
    path = f"output/g5k/consumption/{run['folder']}/data_{run['nb']}.json"
    with Path(path).open("w") as file:
        json.dump(results, file)
