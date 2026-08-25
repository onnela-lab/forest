from dataclasses import dataclass
from enum import Enum


class OSMTags(Enum):
    """ This class enumerates all OSM keys for the """
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


@dataclass
class Hyperparameters:
    """ Class containing hyperparameters for gps imputation and trajectory summary statistics
    calculation.
    
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
