"""Utility functions used by multiple trees
"""

import os

import ratelimit
import requests

from forest.constants import (
    OSM_OVERPASS_URL, OSM_OVERPASS_USER_AGENT, OVERPASS_CALLS_PER_MINUTE
)


def get_ids(study_folder: str) -> list:
    """Get subfolders in directory, excluding registry and hidden folders

    Args:
        study_folder(str): Filepath to the folder containing desired
            subdirectories. Should be an absolute filepath

    Returns:
        List of subdirectories of the study_folder.
    """
    list_of_dirs = []
    for subdir in os.listdir(study_folder):
        is_folder = os.path.isdir(os.path.join(study_folder, subdir))
        if (not subdir.startswith(".")) and subdir != "registry" and is_folder:
            list_of_dirs.append(subdir)
    return list_of_dirs


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=OVERPASS_CALLS_PER_MINUTE, period=60)
def overpass_request_json(query: str, method: str = "GET") -> dict:
    """Run an Overpass query and return the parsed JSON response."""

    method_upper = method.upper()
    if method_upper not in ("GET", "POST"):
        raise ValueError("method must be GET or POST")

    # Overpass accepts the query as "data" in the query string for GET
    # requests and in the body for POST requests
    payload = {"data": query}
    response = requests.request(
        method_upper,
        url=OSM_OVERPASS_URL,
        params=payload if method_upper == "GET" else None,
        data=payload if method_upper == "POST" else None,
        timeout=60,
        headers={"User-Agent": OSM_OVERPASS_USER_AGENT},
    )
    response.raise_for_status()
    return response.json()
