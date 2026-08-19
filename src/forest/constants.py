"""Contains parameters used for API calls in Forest."""

import math
import os
from dataclasses import dataclass
from enum import Enum
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray


#
## Type Shorthands
#

# This module suffers from extremely verbose typing strings that are impossible to read.
# To make it legible we will create a shorthand.
FP = float | np.float64
BoolArr = NDArray[np.bool_]
FP64Arr = NDArray[np.float64]
FPorArr = FP | FP64Arr

# src/forest/jasmine/traj2stats.py
MOBILITY_TRACE_CACHE = dict[tuple[int, int, int] | str, tuple[FP64Arr, FP64Arr, FP64Arr] | float]


#
## Constants
#

## src: SOGP:
SECONDS_PER_DAY_TIMES_PI = 86_400 * math.pi
SECONDS_PER_WEEK_TIMES_PI = 604_800 * math.pi


# src/forest/jasmine/data2mobmat.py
EARTH_RADIUS_METERS = 6.371*10**6
FP_TOLERANCE = 1e-6  # a minimum threshold


## Time
SECONDS_IN_DAY = 60 * 60 * 24

# Openrouteservice API is limited to 40 requests per minute for free accounts
# https://openrouteservice.org/plans/
ORS_API_CALLS_PER_MINUTE = int(os.getenv("FOREST_ORS_API_CALLS_PER_MINUTE", default="40"))
# URL of Openrouteservice instance
ORS_API_BASE_URL = os.getenv("FOREST_ORS_API_BASE_URL", default="https://api.openrouteservice.org")
# URL of OpenStreetMap instance

OSM_OVERPASS_URL = os.getenv(
    "FOREST_OSM_OVERPASS_URL", default="https://overpass-api.de/api/interpreter"
)
# User-Agent for Overpass API requests (required by usage policy)
OSM_OVERPASS_USER_AGENT = os.getenv(
    "FOREST_OSM_OVERPASS_USER_AGENT", default="forest/1.0 (https://github.com/onnela-lab/forest)"
)
# Overpass API asks clients to stay below roughly 10000 requests per day
# https://dev.overpass-api.de/overpass-doc/en/preface/commons.html
OVERPASS_CALLS_PER_MINUTE = int(os.getenv("FOREST_OSM_OVERPASS_CALLS_PER_MINUTE", default="40"))


class Frequency(Enum):
    """This class enumerates possible frequencies for summary data."""
    MINUTE = 1
    HOURLY = 60
    THREE_HOURLY = 3 * 60
    SIX_HOURLY = 6 * 60
    TWELVE_HOURLY = 12 * 60
    DAILY = 24 * 60
    HOURLY_AND_DAILY = -1


class OSMTags(Enum):
    """This class enumerates all OSM keys."""
    AERIALWAY = "aerialway"
    AEROWAY = "aeroway"
    AMENITY = "amenity"
    BARRIER = "barrier"
    BOUNDARY = "boundary"
    BUILDING = "building"
    CRAFT = "craft"
    EMERGENCY = "emergency"
    GEOLOGICAL = "geological"
    HEALTHCARE = "healthcare"
    HIGHWAY = "highway"
    HISTORIC = "historic"
    LANDUSE = "landuse"
    LEISURE = "leisure"
    MAN_MADE = "man_made"
    MILITARY = "military"
    NATURAL = "natural"
    OFFICE = "office"
    PLACE = "place"
    POWER = "power"
    PUBLIC_TRANSPORT = "public_transport"
    RAILWAY = "railway"
    ROUTE = "route"
    SHOP = "shop"
    SPORT = "sport"
    TELECOM = "telecom"
    TOURISM = "tourism"
    WATER = "water"
    WATERWAY = "waterway"


## src: simulate_gps_data.py


ACTIVE_STATUS_LIST = range(11)  # ranges are special objects that can be iterated over multiple times
TRAVELLING_STATUS_LIST = range(11)


## Numba type declaration for the great circle functions
# numba has its own type system, we use some of them here
REQUIRES_FOUR_FLOATS = "float64[:](float64, float64, float64, float64)"


# src/forest/jasmine/traj2stats.py
COORDS_OUT_OF_RANGE_MSG = "Trajectory coordinates are not in the range of [-90, 90] and [-180, 180]."


@dataclass
class Hyperparameters:
    """Class containing hyperparemeters for gps imputation and trajectory
     summary statistics calculation.
    
    Args:
        itrvl, accuracylim, r, w, h: hyperparameters for the gps_to_mobmat function.
        
        itrvl, r: hyperparameters for the infer_mobmat function.
        
        l1, l2, l3, a1, a2, b1, b2, b3, sigma2, tol, d: hyperparameters for the bv_select function.
        
        l1, l2, a1, a2, b1, b2, b3, g, method, switch, num, linearity: hyperparameters for the
            impute_gps function.
        
        itrvl, r, w, h: hyperparameters for the imp_to_traj function.
        
        log_threshold: int, time spent in a pause needs to exceed the
            log_threshold to be placed in the log only if save_osm_log True, in minutes
        
        split_day_night: bool, True if you want to split all metrics to datetime and nighttime
            patterns only for daily frequency
        
        person_point_radius: float, radius of the person's circle when discovering places near him
            in pauses
        
        place_point_radius: float, radius of place's circle when place is returned as centre
            coordinates from osm
            
        save_osm_log: bool, True if you want to output a log of locations visited and their tags
        
        quality_threshold: float, a percentage value of the fraction of data
            required for a summary to be created
        
        pcr_bool: bool, True if you want to calculate the physical circadian rhythm
        
        pcr_window: int, number of days to look back and forward for calculating the physical
            circadian rhythm
        
        pcr_sample_rate: int, number of seconds between each sample for calculating the physical
            circadian rhythm
    """
    # imputation hyperparameters
    l1: int = 60 * 60 * 24 * 10
    l2: int = 60 * 60 * 24 * 30
    l3: float = 0.002
    g: int = 200
    a1: int = 5
    a2: int = 1
    b1: float = 0.3
    b2: float = 0.2
    b3: float = 0.5
    d: int = 100
    sigma2: float = 0.01
    tol: float = 0.05
    switch: int = 3
    num: int = 10
    linearity: int = 2
    method: str = "GLC"
    itrvl: int = 10
    accuracylim: int = 51
    r: float | None = None
    w: float | None = None
    h: float | None = None
    
    # summary statistics hyperparameters
    save_osm_log: bool = False
    log_threshold: int = 60
    split_day_night: bool = False
    person_point_radius: float = 2
    place_point_radius: float = 7.5
    quality_threshold: float = 0.05
    pcr_bool: bool = False
    pcr_window: int = 14
    pcr_sample_rate: int = 30


## Constants for working with Beiwe time formats.

# seconds
MIN_S = 60
HOUR_S = 60 * MIN_S
DAY_S = 24 * HOUR_S

# milliseconds
SEC_MS = 1000
MIN_MS = 1000 * MIN_S
HOUR_MS = 1000 * HOUR_S
DAY_MS = 1000 * DAY_S
WEEK_MS = 7 * DAY_MS
YEAR_MS = 365 * DAY_MS
TIME_MS = dict(
    zip(
        ["milliseconds", "seconds", "minutes", "hours", "days", "weeks", "years"],
        [1, SEC_MS, MIN_MS, HOUR_MS, DAY_MS, WEEK_MS, YEAR_MS],
    )
)

# Beiwe day order
DAY_ORDER = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]

# Beiwe time formats
DATA_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"  # used in raw Beiwe data
FILENAME_DATETIME_FORMAT = "%Y-%m-%d %H_%M_%S"  # used in raw Beiwe file names

# human-readable time formats
DATE_FORMAT = "%Y-%m-%d"  # ISO 8601 date
TIME_FORMAT = "%H:%M:%S"  # ISO 8601 time
TIMEZONE_FORMAT = "%Z"  # timezone name
OFFSET_FORMAT = "%z"  # UTC offset

# human-readable and RFC 3339-compliant
NAIVE_DATETIME_FORMAT = " ".join([DATE_FORMAT, TIME_FORMAT])
AWARE_DATETIME_FORMAT = " ".join([DATE_FORMAT, TIME_FORMAT, TIMEZONE_FORMAT])
OFFSET_DATETIME_FORMAT = " ".join([DATE_FORMAT, TIME_FORMAT, OFFSET_FORMAT])

# commonly used time zones
UTC = ZoneInfo("UTC")
