"""Contains parameters used for API calls in Forest."""

import os

# 1.5 seconds delay between each call for ORS API
# since the API is limited to 40 requests per minute for free accounts
ORS_API_DELAY = float(os.getenv("FOREST_ORS_API_DELAY", default="1.5"))
# in case of local ORS instance
ORS_API_BASE_URL = os.getenv("FOREST_ORS_BASE_URL",
                             default="https://api.openrouteservice.org")
# in case of local OSM instance
OSM_OVERPASS_URL = os.getenv("FOREST_OSM_OVERPASS_URL",
                             default="https://overpass-api.de/api/interpreter")
