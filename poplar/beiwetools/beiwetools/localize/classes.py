'''Tools for working with localized summaries and metrics.

- LocalFollowup is a simple class for representing summaries generated for
uniformly-spaced windows.
- LocalFollowup is designed to track summaries for uniformly-sized windows,
e.g. minutes, hours, or days.  Window size is fixed for a LocalFollowup object.
For tracking windows of different sizes, use multiple LocalFollowup instances.
- Summaries are stored in files which can be updated with the csv_to_followup()
method.
- Interact with stored summaries using a LocalFollowup class.  Summary files 
are only opened to update contents or to assemble data sets.
- By default, summaries are stored in pickled pandas.Series objects
with dtype = 'category'.  This is a safe dtype for floats, ints, bools, and
missing values (np.NaN or None).  
- If storage is an issue, use dtype = 'float' for greater efficiency (~35%
reduction).  Note that 'float' will coerce integers to floats, and booleans to
float values (0.0 / 0.1).

'''
import os
from logging import getLogger
import numpy as np
import pandas as pd

from beiwetools.helpers.functions import (setup_directories, 
                                          write_json, read_json)
from beiwetools.helpers.time_constants import day_ms, time_only, UTC
from beiwetools.helpers.time import between_days, next_day, to_readable


logger = getLogger(__name__)


class LocalFollowup():
    '''
    Attributes:
        path (str): Path to directory where localized data are stored.
        days (list): Inclusive list of local dates (str) in '%Y-%m-%d' format.
        window (int): Duration of tracked windows in milliseconds.
        window_times (list): Starting times (str) for each window in 
            '%H:%M:%S' format.
        variables (dict): Keys are names of variables that are tracked.
            Values are dtypes for pandas.Series, e.g. 'float' or 'category'.
    '''
    def __init__():
        pass
    
    @classmethod
    def create(cls, output_dir, name, start_date, end_date, window):
        '''
        Initialize an instance, create directories & files.
        
        Args:
            output_dir (str): Where to create a storage directory.
            name (str): Name for storage directory.
            start_date (str): Starting date in '%Y-%m-%d' format.
            end_date (str): Ending date in '%Y-%m-%d' format.
            window (int): Length of windows in milliseconds.
                Use MIN_MS = 60000 for minute-by-minute summaries.
                Use DAY_MS = 86400000 for daily summaries.
            
        Returns:
            self (LocalFollowup): A new LocalFollowup instance.

        '''
        if day_ms % window != 0:
            logger.warning('Window duration doesn\'t evenly divide into days.')
        else:
            self = cls.__new__(cls)           
            # storage directory
            self.path = os.path.join(output_dir, name)
            # get dates in range
            self.days = between_days(start_date, end_date)        
            # get window times
            self.window = window
            window_ms = np.arange(0, day_ms, window)
            self.window_times = [to_readable(t, time_only, to_tz = UTC) \
                                 for t in window_ms]            
            # initialize empty variable dictionary
            self.variables = {}
            # initialize directories
            records_dir = os.path.join(self.path, 'records')
            day_dir = os.path.join(self.path, 'days')
            date_dirs = [os.path.join(day_dir, d) for d in self.days]            
            setup_directories([records_dir, day_dir] + date_dirs)                 
            # save attributes
            parameters = dict(zip(['path', 'days', 'window', 
                                   'window_times', 'variables'],
                                  [self.path, self.days, self.window,
                                   self.window_times, self.variables]))
            write_json(parameters, 'parameters', records_dir)
            return(self)
        
    @classmethod
    def load(cls, dirpath):
        '''
        Load a directory of followup data into a LocalFollowup instance.

        Args:
            dirpath (str):  Location of an exported LocalFollowup.

        Returns:
            self (LocalFollowup): A new LocalFollowup instance.
        '''
        try:
            self = cls.__new__(cls)           
            parameters_file = os.path.join(dirpath, 'records', 
                                           'parameters.json')
            self.__dict__ = read_json(parameters_file)
            return(self)            
        except:
            logger.warning('Unable to load followup data.')




        
        # a = pd.Series([1, 2, 3, None], 
        #               dtype = 'category')

        # b = pd.Series([1, 2, 3, None], 
        #               dtype = 'float')

        # a.to_pickle(os.path.join(output_dir, 'a4'))
        # b.to_pickle(os.path.join(output_dir, 'b4'))

        # a_pickle = pd.read_pickle(os.path.join(output_dir, 'a'))

        
        # output_dir = '/home/josh/Desktop/test/'
        # name = 'test_followup'
        # start_date = '2020-09-01'
        # end_date = '2020-10-07'
        # window = 60000
        
    
    
        
    def add_variables(self, new_variables):
        '''
        Create placeholders for variables to track.
        
        Args:
            new_variables (list): List of tuples describing new variables.
                Tuples are pairs of strings, (<variable name>,<variable type>).
                Can also just be a <variable name> (str), in which case the 
                <variable_type> will be 'category'.

        Returns:
            None
        '''
        for v in new_variables:
            if isinstance(v, str):
                v = (v, 'category')
            variable_name = v[0]
            variable_type = v[1]
            # update records
            self.variables[variable_name] = variable_type
            # create placeholders for values
            for date in self.days:
                series_filepath = os.path.join(self.path, 'days', 
                                               date, variable_name)
                if not os.path.exists(series_filepath):
                    values = [None]*len(self.window_times)                
                    index = zip([date]*len(self.window_times), 
                                self.window_times)
                    date_series = pd.Series(data = values, index = index,
                                            dtype = variable_type)
                    date_series.to_pickle(series_filepath)
                            
    def extend_followup(self, new_end_date):
        '''
        Extend the followup period to a later end date.
        '''
        # get new date range
        last_date = self.days[-1]
        next_date = next_day(last_date)
        new_dates = between_days(next_date, new_end_date)
        # create folders
        new_date_dirs = [os.path.join(self.path, 'days', d) for d in new_dates]
        setup_directories(new_date_dirs)
        # initialize variables
        self.add_variables(list(self.variables.keys()))        
        # update records
        self.days += new_dates
        
    def get_long(self, variable, start_date, end_date):
        pass
        
    def get_wide(self, variable, start_date, end_date):
        pass
        

def csv_to_followup(filepath, followup, column_label, variable):
    '''
    
    
    Args:
        filepath (str): Path to a csv file with a timestamp header.        
        
    Returns:
        None
    
    '''    
    
    pass
    