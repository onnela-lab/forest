'''Functions for working with raw Beiwe accelerometer data.

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
        df (DataFrame): A pandas dataframe containing raw Beiwe accelerometer 
            data from an iPhone.
    
    Returns:
        None
    '''
    for c in df.columns:
        if c in ['x', 'y', 'z']:
            df[c] = G * df[c]


def read_acc(path, device_os, keep = raw_header,
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
    if device_os == 'iOS':
        gee_to_mps2(df)
    return(df)
    

# Some data frame functions:
def overall_dynamic_body_acceleration(df, win_width_s): pass
def vector_magnitude(df): return((df.x**2 + df.y**2 + df.z**2)**(0.5))
def euclidean_norm_minus_one(df): return(np.maximum(df.vecmag - G, 0))
def amplitude_deviation(df): 
    r_bar = np.mean(df.vecmag)
    return(np.abs(df.vecmag - r_bar))
def sum_of_amplitudes(df): return(df.x + df.y + df.z)
def proportion_perpendicular_to_z(df): return(1 - np.abs(df.z)/df.vecmag)
def z_minus_one(df): return(np.abs(np.abs(df.z) - G))
    
# And the corresponding transformation objects:
vecmag = Transformation('vecmag', 'Vector Magnitude', 'm/s^2', 
                        'accelerometer', vector_magnitude)
enmo   = Transformation('enmo',   'Euclidean Norm Minus One', 
                        'm/s^2', 'accelerometer', euclidean_norm_minus_one)
ampdev = Transformation('ampdev', 'Amplitude Deviation', 
                        'm/s^2', 'accelerometer', amplitude_deviation)
sumamp = Transformation('sumamp', 'Sum of Amplitudes', 'm/s^2', 
                        'accelerometer', sum_of_amplitudes)
ppz    = Transformation('ppz', 
                        'Proportion of Acceleration Perpendicular to Z-Axis',
                        '', 'accelerometer', proportion_perpendicular_to_z)
zmo    = Transformation('zmo', 'Z-Axis Acceleration Minus One',
                        'm/s^2', 'accelerometer', z_minus_one)
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
     'ppz':    ppz,
     'zmo':    zmo,
     'x': x,
     'y': y,
     'z': z
     })
    

def absolute_activity_index(df, ref): 
    '''
    Bai et al, 2016.

    Args:
        df (dataframe): Dataframe of accelerometer data.
        ref (float): The sum of variances  calculated from a reference period:
            var(x) + var(y) + var(z)
    '''
    temp = np.sum([np.var(df[a], ddof=1) for a in ['x', 'y', 'z']])
    diff = (temp - ref)/3
    return(np.sqrt(np.max([diff, 0])))


def relative_activity_index(df, ref): 
    '''
    Bai et al, 2016.

    Args:
        df (dataframe): Dataframe of accelerometer data.
        ref (float): The sum of variances calculated from a reference period:
            var(x) + var(y) + var(z)
    '''
    temp = np.sum([np.var(df[a], ddof=1) for a in ['x', 'y', 'z']])
    diff = (temp/ref - 1)/3
    return(np.sqrt(np.max([diff, 0])))


def absolute_activity_range_index(df, ref): 
    '''
    Args:
        df (dataframe): Dataframe of accelerometer data.
        ref (float): Sum of interquartile ranges calculated from a reference 
            period: iqr(x) + range(y) + range(z)
                
    '''
    temp = np.sum([np.max(df[a])-np.min(df[a]) for a in ['x', 'y', 'z']])
    diff = (temp - ref)/3
    return(np.max([diff, 0]))


def relative_activity_range_index(df, ref): 
    '''
    Args:
        df (dataframe): Dataframe of accelerometer data.
        ref (float): Sum of interquartile ranges calculated from a reference 
            period: iqr(x) + range(y) + range(z)
                
    '''
    temp = np.sum([np.max(df[a])-np.min(df[a]) for a in ['x', 'y', 'z']])
    diff = (temp/ref - 1)/3
    return(np.max([diff, 0]))


metrics = {
    'ai':   absolute_activity_index,
    'ai0':  relative_activity_index,
    'ari':  absolute_activity_range_index,
    'ari0': relative_activity_range_index
    }