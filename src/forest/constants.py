""" Contains numerous constants used throughout the forest package. """

import math
import os
from enum import Enum
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numpy.typing import NDArray


#
## Type Shorthands
#

# Some modules suffer from extremely verbose Python typing strings that are impossible to read.
# To make it legible we will create a shorthand.
FP = float | np.float64
BoolArr = NDArray[np.bool_]
FP64Arr = NDArray[np.float64]
FPorArr = FP | FP64Arr

# src/forest/jasmine/traj2stats.py
MOBILITY_TRACE_CACHE = dict[tuple[int, int, int] | str, tuple[FP64Arr, FP64Arr, FP64Arr] | float]

K0_PARAMS = tuple[int, int, float, int, int, float, float, float]
K1_PARAMS = tuple[int, int, int, int, float, float, float, int]

# Numba (an optimization library) has special type declarations that we use in Jasmine's great
# circle functions. These have a special Numba library purpose and are not recognized by Python.
REQUIRES_FOUR_FLOATS = "float64[:](float64, float64, float64, float64)"


#
## Constants
#

class Frequency(Enum):
    """ Frequency options available for summary data output.
    These options are used in the frequency parameters of various functions across Forest.
    
    Values outside of these options may not be supported and may result in errors or unexpected
    behavior. """
    
    MINUTE = 1
    HOURLY = 60
    THREE_HOURLY = 3 * 60
    SIX_HOURLY = 6 * 60
    TWELVE_HOURLY = 12 * 60
    DAILY = 24 * 60
    HOURLY_AND_DAILY = -1


# The one true timezone
UTC = ZoneInfo("UTC")


## Numerical constants

FP_TOLERANCE = 1e-6  # a minimum threshold (Jasmine)

SECONDS_IN_DAY = 60 * 60 * 24

SECONDS_PER_DAY_TIMES_PI = 86_400 * math.pi  ## src: SOGP
SECONDS_PER_WEEK_TIMES_PI = 604_800 * math.pi

EARTH_RADIUS_METERS = 6.371*10**6  # src: data2mobmat
GEE = 9.80665  # Earth's gravity (meters/second^2)

# URLs and other constants

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


## Named Hardcoded Ranges
# ranges are special objects that can be iterated over multiple times

ACTIVE_STATUS_LIST = range(11)
TRAVELLING_STATUS_LIST = range(11)


## Constants for working with Beiwe time formats

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


## Beiwe time formats

DATA_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"  # used in raw Beiwe data
FILENAME_DATETIME_FORMAT = "%Y-%m-%d %H_%M_%S"  # used in raw Beiwe file names

# human-readable time formats
DATE_FORMAT = "%Y-%m-%d"  # ISO 8601 date
TIME_FORMAT = "%H:%M:%S"  # ISO 8601 time
TIMEZONE_FORMAT = "%Z"  # timezone name
OFFSET_FORMAT = "%z"  # UTC offset

# human-readable and RFC 3339-compliant
NAIVE_DATETIME_FORMAT = f"{DATE_FORMAT} {TIME_FORMAT}"
AWARE_DATETIME_FORMAT = f"{DATE_FORMAT} {TIME_FORMAT} {TIMEZONE_FORMAT}"
OFFSET_DATETIME_FORMAT = f"{DATE_FORMAT} {TIME_FORMAT} {OFFSET_FORMAT}"

# Bytes, see https://physics.nist.gov/cuu/Units/binary.html
BYTES_DEC = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "PB": 10**15,
}
BYTES_BIN = {
    "B": 1,
    "KiB": 2**10,
    "MiB": 2**20,
    "GiB": 2**30,
    "TiB": 2**40,
    "PiB": 2**50,
}
BYTES = {**BYTES_DEC, **BYTES_BIN}


## Survey constants

# We want our default date to be farther in the past than any Beiwe data could have been collected,
# so we never cut off data by default. But, if we set our default date too far in the past, we would
# generate too many weekly survey timings
EARLIEST_DATE = "2010-01-01"

# load events & question types dictionary
QUESTION_TYPES_LOOKUP = {
    "Android":
        {
            "Checkbox Question": "checkbox",
            "Info Text Box": "info_text_box",
            "Open Response Question": "free_response",
            "Radio Button Question": "radio_button",
            "Slider Question": "slider",
        },
    "iOS":
        {
            "checkbox": "checkbox",
            "free_response": "free_response",
            "info_text_box": "info_text_box",
            "radio_button": "radio_button",
            "slider": "slider",
        }
}

# On 6 Dec 2016, a commit was pushed which changed the behavior of Android Radio question answers.
# The commit was called "Gets nullable Integer answers from sliders and radio button questions" and
# can be found at.

# https://github.com/onnela-lab/beiwe-android/commit/6341eb5498ceeffcb64d65c2dd2bcfdab9b982f2
ANDROID_NULLABLE_ANSWER_CHANGE_DATE = pd.to_datetime("2016-12-06")

#
## Data Columns - Collect lists of columns for data types in one place
#

CALLS_COLS = [
    "timestamp",
    "UTC time",
    "hashed phone number",
    "call type",
    "duration in seconds",
]

TEXTS_COLS = [
    "timestamp",
    "UTC time",
    "hashed phone number",
    "sent vs received",
    "message length",
    "time sent",
]


#
## Messages
#

COORDS_OUT_OF_RANGE_MSG = "Trajectory coordinates are not in the range of [-90, 90] and [-180, 180]."

NO_SURVEY_HISTORY_MSG = "No survey history path included. If you have changed radio survey " \
"answer choices since starting your study, and if you used semicolons or commas in those " \
"answer choices, incorrect survey responses may be output for android devices"
