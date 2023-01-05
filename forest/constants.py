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
    HOURLY = 1
    DAILY = 24
    HOURLY_AND_DAILY = "hourly_and_daily"
    THREE_HOURLY = 3
    SIX_HOURLY = 6
    TWELVE_HOURLY = 12


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
