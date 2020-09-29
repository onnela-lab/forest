'''Functions for working with raw Beiwe GPS data.

'''
from logging import getLogger
import pandas as pd
from math import pi, sin, acos, sqrt
from .headers import raw_header
from beiwetools.helpers.process import clean_dataframe


logger = getLogger(__name__)


# mean radius of the Earth
R = 6.371*10**6


def read_gps(filepath, keep = raw_header,
             accuracy_cutoff = None,
             clean_args = (True, True, True)):
    '''
    Read a raw Beiwe GPS file.  
    Optionally, drop inaccurate observations.
    
    Args:
        filepath (str): Path to the file.
        keep (list): Which columns to keep.
        accuracy_cutoff (float or Nonetype):
            Observations with accuracy >= cutoff are dropped.
            If None, then no observations are dropped.
        clean_args (tuple): Args for clean_dataframe().

    Returns:
        df (DataFrame): Pandas dataframe of GPS data.    
    '''    
    df = pd.read_csv(filepath, usecols = keep)
    clean_dataframe(df, *clean_args)
    # drop high accuracy obsevations
    if not accuracy_cutoff is None:
        df = df.loc[df.accuracy < accuracy_cutoff]
        clean_dataframe(df, *clean_args)
    return(df)


def gps_project(df, location_ranges, action = 'replace',
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
   
    
def get_displacement(first, last):
    '''
    Get displacements from projected GPS data.
    
    Args:
        first, last:  Rows from a pandas dataframe that has columns 'timestamp', 
            'x', 'y', 'z'.
    
    Returns:
        delta_x (float): North/South displacement (meters).
        delta_y (float): East/West displacement (meters).
        delta_z (float): Up/Down displacement (meters).
        seconds (float): Elapsed seconds. 
        xy_distance (float): Norm of (delta_x, delta_y).
        xyz_distance (float): Norm of (delta_x, delta_y, delta_z).
    '''
    delta_x = last.x - first.x
    delta_y = last.y - first.y
    delta_z = last.z - first.z
    seconds = (last.timestamp - first.timestamp)/1000
    xy_distance = sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    xyz_distance = sqrt(delta_x**2 + delta_y**2)
    return([delta_x, delta_y, delta_z, seconds, xy_distance, xyz_distance])



