"""Contains parameters used for API calls in Forest."""

from enum import Enum
import os

# Openrouteservice API is limited to 40 requests per minute for free accounts
# https://openrouteservice.org/plans/
ORS_API_CALLS_PER_MINUTE = int(os.getenv("FOREST_ORS_API_CALLS_PER_MINUTE",
                                         default="40"))
# URL of Openrouteservice instance
ORS_API_BASE_URL = os.getenv("FOREST_ORS_API_BASE_URL",
                             default="https://api.openrouteservice.org")
# URL of OpenStreetMap instance
OSM_OVERPASS_URL = os.getenv("FOREST_OSM_OVERPASS_URL",
                             default="https://overpass-api.de/api/interpreter")


class Frequency(Enum):
    """This class enumerates possible frequencies for summary data."""
    HOURLY = "hourly"
    DAILY = "daily"
    BOTH = "both"
