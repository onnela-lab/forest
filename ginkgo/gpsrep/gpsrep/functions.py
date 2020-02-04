'''
Functions for working with raw Beiwe GPS data.
'''
import logging
import pandas as pd
from math import pi, sin, acos
from .headers import raw_header
from beiwetools.helpers.process import clean_dataframe


logger = logging.getLogger(__name__)


# mean radius of the Earth
R = 6.371*10**6


def read_gps(path, keep = raw_header,
             accuracy_cutoff = None,
             location_ranges = None,
             clean_args = (True, True, True)):
    '''
    Read a raw Beiwe GPS file.  
    Optionally, drop inaccurate observations and project to 2D plane.
    
    Args:
        path (str): Path to the file.
        keep (list): Which columns to keep.
            Note that projections will replace lat/long/alt with x/y/z.
        accuracy_cutoff (float or Nonetype):
            Observations with accuracy >= cutoff are dropped.
            If None, then no observations are dropped.
        location_ranges (Nonetype or dict):
            If None, returns dataframe with latitude, longitude, altitude.
            Otherwise, performs projection and returns dataframe with x, y, z.
            Keys are 'latitude', 'longitude', 'altitude'.
            Values are pairs [min, max].
        clean_args (tuple): Args for clean_dataframe().

    Returns:
        df (DataFrame): Pandas dataframe of GPS data.    
    '''    
    df = pd.read_csv(path, usecols = keep)
    clean_dataframe(df, *clean_args)
    # drop high accuracy obsevations
    if not accuracy_cutoff is None:
        df = df.loc[df.accuracy < accuracy_cutoff]
        clean_dataframe(df, *clean_args)
    # project to 2D plane
    if not location_ranges is None:
        project_gps(df, location_ranges, action = 'replace')
    return(df)


def project_gps(df, location_ranges, action = 'replace',
                coordinates = ['latitude', 'longitude', 'altitude']):
    '''
    Project latitude and longitude to a 2-D surface.
    Shifts altitude so that the minimum observed altitude is 0.
    Adapted from the function LatLong2XY(Lat,Lon) found here:
		forest/sweet_osmanthus/imputeGPS.py 

    Args:
        df (DataFrame): Dataframe with latitude and longitude observations.
        location_ranges (dict): 
            Keys are 'latitude', 'longitude', 'altitude'.
            Values are pairs [min, max].
        action (str): 
            If 'append', adds x, y, z columns to df.
            If 'replace', replaces latitude, longitude, altitude in df with x, y, z.
            If 'new', returns a new dataframe with only columns timestamp, x, y, z.
        coordinates (list): Names of the latitude, longitude, and altitude columns in df.
        
    Returns:
        new_df (DataFrame): 
            If action == 'new', returns a new dataframe with columns timestamp, x, y, z.
            Otherwise, returns None.
    '''
    latitude_rads  = df[coordinates[0]].to_numpy() * (pi/180)
    longitude_rads = df[coordinates[1]].to_numpy() * (pi/180)
    lam_min, lam_max = [r*(pi/180) for r in location_ranges['latitude']]
    phi_min, phi_max = [r*(pi/180) for r in location_ranges['longitude']]
    d1 = (lam_max - lam_min) * R
    d2 = (phi_max - phi_min) * R * sin(pi/2 - lam_max)
    d3 = (phi_max - phi_min) * R * sin(pi/2 - lam_min)
    w1 = (latitude_rads -  lam_min) / (lam_max - lam_min)
    w2 = (longitude_rads - phi_min) / (phi_max - phi_min)
    x = w1*(d3 - d2)/2 + w2*(d3*(1 - w1) + d2*w1)
    y = w1*d1*sin(acos((d3 - d2)/(2*d1)))
    z = df[coordinates[2]].to_numpy() - location_ranges['altitude'][0]
    if action == 'append':
        df['x'] = x       
        df['y'] = y
        df['z'] = z
    elif action == 'replace': 
        df.rename(columns = {'latitude' : 'x', 
                             'longitude': 'y', 
                             'altitude' : 'z'}, inplace = True) 
        df['x'] = x
        df['y'] = y
        df['z'] = z
    elif action == 'new':
        new_df = pd.DataFrame({'timestamp': df.timestamp, 'x':x, 'y':y, 'z':z})
        return(new_df)
    





