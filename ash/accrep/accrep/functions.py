'''Functions for working with raw Beiwe accelerometer data

'''
import logging
import numpy as np
import pandas as pd

from collections import OrderedDict

from .headers import raw_header
from beiwetools.helpers.classes import Transformation
from beiwetools.helpers.process import clean_dataframe


logger = logging.getLogger(__name__)


# gravitational constant
G = 9.80665


def gee_to_mps2(df):
    '''
    iPhones report acceleration in units of g.
    This function converts to units of m/s^2, as used by Android.
    Conversion is done in place.
    
    Args:
        df (DataFrame): 
            A pandas dataframe containing raw Beiwe accelerometer data from an iPhone.
    
    Returns:
        None
    '''
    for c in df.columns:
        if c in ['x', 'y', 'z']:
            df[c] = G * df[c]


def read_acc(path, os, keep = raw_header,
             clean_args = (True, True, True)):
    '''
    Open a raw Beiwe accelerometer file.
    Convert to m/s^2 if file is from an iPhone.
    
    Args:
        path (str): Path to the file.
        os (str): 'Android' or 'iOS'
        keep (list): Which columns to keep.
        clean_args (tuple): Args for clean_dataframe()
        
    Returns:
        df (DataFrame): Pandas dataframe of accelerometer data.
    '''
    df = pd.read_csv(path, usecols = keep)
    clean_dataframe(df, *clean_args)
    if os == 'iOS':
        gee_to_mps2(df)
    return(df)
    

# Some data frame functions:
def vector_magnitude(df): return((df.x**2 + df.y**2 + df.z**2)**(0.5))
def euclidean_norm_minus_one(df): return(np.max(vector_magnitude(df) - G, 0))
def amplitude_deviation(df): 
    r = vector_magnitude(df)
    r_bar = np.mean(r)    
    return(np.abs(r - r_bar))
def sum_of_amplitudes(df): return(df.x + df.y + df.z)    

# To add:
# def overall_dynamic_body_acceleration(df, win_width_s): pass
# def absolute_activity_index(df, ): pass
# def relative_activity_index(df, ): pass

# And the corresponding transformation objects:
vecmag = Transformation('vecmag', 'Vector Magnitude', 'm/s^2', 'accelerometer', vector_magnitude)
enmo   = Transformation('enmo',   'Euclidean Norm Minus One', 'm/s^2', 'accelerometer', euclidean_norm_minus_one)
ampdev = Transformation('ampdev', 'Amplitude Deviation', 'm/s^2', 'accelerometer', amplitude_deviation)
sumamp = Transformation('sumamp', 'Sum of Amplitudes', 'm/s^2', 'accelerometer', sum_of_amplitudes)
# Dummy transformations for axes:
x = Transformation('x', 'X-Axis Acceleration', 'm/s^2', 'accelerometer', None)
y = Transformation('y', 'Y-Axis Acceleration', 'm/s^2', 'accelerometer', None)
z = Transformation('z', 'Z-Axis Acceleration', 'm/s^2', 'accelerometer', None)

# Put data frame functions into a dictionary:
transformations = OrderedDict(
    {'vecmag': vecmag,
     'enmo':   enmo,
     'ampdev': ampdev,
     'sumamp': sumamp,
     'x': x,
     'y': y,
     'z': z
     })