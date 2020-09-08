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
- Summaries are stored in pickled pandas.Series objects with dtype = 'float'.  
- Note that integers will be coerced to floats, booleans to 0.0 / 0.1, and 
None to np.nan.

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
        variables (list): Names of variables that are tracked.

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
            self.variables = []
            # initialize directories
            records_dir = os.path.join(self.path, 'records')
            day_dir = os.path.join(self.path, 'days')
            date_dirs = [os.path.join(day_dir, d) for d in self.days]            
            setup_directories([records_dir, day_dir] + date_dirs)                 
            # save attributes
            self.save_attributes()
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

    def save_attributes(self):
        '''
        Update attribute records.
        '''
        parameters = dict(zip(['path', 'days', 'window', 
                               'window_times', 'variables'],
                              [self.path, self.days, self.window,
                               self.window_times, self.variables]))
        records_dir = os.path.join(self.path, 'records')
        write_json(parameters, 'parameters', records_dir)

    def add_variables(self, new_variables):
        '''
        Create placeholders for variables to track.
        
        Args:
            new_variables (list): List of variable names.

        Returns:
            None
        '''
        for variable_name in new_variables:
            # update records
            self.variables.append(variable_name)
            # create placeholders for values
            for date in self.days:
                series_filepath = os.path.join(self.path, 'days', 
                                               date, variable_name)
                if not os.path.exists(series_filepath):
                    values = [None]*len(self.window_times)                
                    index = zip([date]*len(self.window_times), 
                                self.window_times)
                    date_series = pd.Series(data = values, index = index,
                                            dtype = 'float')
                    date_series.to_pickle(series_filepath)
        # update attributes
        self.save_attributes()
                            
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
        # update attributes
        self.save_attributes()
        
    def get_long(self, variable, return_as,
                 start_date = None, end_date = None):
        '''
        Get a single 1D-array or Series with one variable's values 
        for a time range.
        
        Args:
            variable (str): An item from self.variables.
            return_as (str): Either 'array' or 'series'.
            start_date (str or Nonetype): Start date for variable.  If None,
                then start date will be self.days[0].
            end_date (str or Nonetype): End date for variable.  If None,
                then end date will be self.days[-1].
            
        Returns:
            long (numpy.ndarray or pandas.Series): Vertically stacked values 
                for the variable.
        '''
        if start_date is None: start_date = self.days[0]
        if end_date is None: end_date = self.days[-1]
        # get date range
        dates_to_return = between_days(start_date, end_date)
        # get values for each day
        paths_to_unpickle = [os.path.join(self.path, 'days', d, variable)\
                             for d in dates_to_return]
        data_list = [pd.read_pickle(p) for p in paths_to_unpickle]
        # assemble values
        long = pd.concat(data_list)
        if return_as == 'array': long = long.to_numpy()
        return(long)

    def get_wide(self, variable, return_as,
                 start_date = None, end_date = None):
        '''
        Get a 2D-array or DataFrame with one variable's values for a 
        time range.  Columns are days; rows are windows.
        
        Args:
            variable (str): An item from self.variables.
            return_as (str): Either 'array' or 'dataframe'.
            start_date (str or Nonetype): Start date for variable.  If None,
                then start date will be self.days[0].
            end_date (str or Nonetype): End date for variable.  If None,
                then end date will be self.days[-1].
            
        Returns:
            wide (numpy.ndarray or pandas.DataFrame): Horizontally stacked 
                values for the variable.
        '''
        if start_date is None: start_date = self.days[0]
        if end_date is None: end_date = self.days[-1]
        # get date range
        dates_to_return = between_days(start_date, end_date)
        # get values for each day
        paths_to_unpickle = [os.path.join(self.path, 'days', d, variable)\
                             for d in dates_to_return]
        data_list = [pd.read_pickle(p) for p in paths_to_unpickle]
        data_dict = dict(zip(dates_to_return, data_list))
        for s in data_list: s.index = self.window_times
        # assemble values
        wide = pd.DataFrame(data_dict)
        if return_as == 'array': wide = wide.to_numpy()
        return(wide)






        
        # output_dir = '/home/josh/Desktop/test/'
        # name = 'test_followup'
        # start_date = '2020-09-01'
        # end_date = '2020-10-07'
        # window = 60000
        


        
        

def csv_to_followup(filepath, followup, offset_history,
                    column_label, variable,
                    f = None, **f_kwargs):
    '''
    Read values from a CSV file into a LocalFollowup object.
    The CSV file must contain a 'timestamp' column containing millisecond
    timestamps.    
    
    Args:
        filepath (str): Path to a CSV file with a timestamp header.        
        followup (LocalFollowup): A LocalFollowup instance.

        offset_history (


        column_label (str): A column label from the CSV file containing values
            to read.
        variable (str): A variable name in followup.variables to store values.
        f (function): An optional function for transforming data after 
            opening with pd.read_csv().  Should take a pandas DataFrame as 
            input and return a pandas DataFrame.
        f_kwargs (dict): Keyword arguments for f.
        
    Returns:
        None
    
    '''    
    data = pd.read_csv(filepath)
    if not f is None:
        data = f(data)    
    timestamps = data['timestamps']
    values = data[column_label]
    date_time_tuples = [('','') for t in timestamps]
    
    to_readable('', to_tz = UTC)
    pass    
    