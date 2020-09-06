'''Classes for working with localized summaries and metrics.

LocalFollowup is a simple class for representing summaries generated for
uniformly-spaced windows.

'''
import os
from logging import getLogger

from beiwetools.helpers.functions import setup_directories
from beiwetools.helpers.time import between_days


logger = getLogger(__name__)


class LocalFollowup():
    '''
    Attributes:
        path (str): Path to directory where localized data are stored.
        days (list): Inclusive list of local dates (str) in '%Y-%m-%d' format.
        window (int): Duration of tracked windows in milliseconds.
        window_times (list): Starting times for each window in 
        variables (list): List of names (str) for variables that are tracked.
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
            end_date (str): Ending date in `%Y-%m-%d` format.
            window (int): Length of windows in milliseconds.
                Use MIN_MS = 6000 for minute-by-minute summaries.
                Use DAY_MS = 86400000 for daily summaries.
            
        Returns:
            None

        '''
        # storage directory
        self.path = os.path.join(output_dir, name)
        # get dates in range
        self.days = between_days(start_date, end_date)        
        # get window times
        self.window = window
        
        output_dir = '/home/josh/Desktop/test/'
        name = 'test_followup'
        start_date = '2020-09-01'
        end_date = '2020-10-07'
        window = 6000
        
    
    @classmethod
    def load(cls, dirpath):

        
    def add_variables(self, variables):
        '''
        Create placeholders for variables to track.
        '''

    def update():
        '''
        Read window values for variables and write to file.
        '''
        
    def extend_followup(new_end_date):
        
    def get_long(variable, start_date, end_date):
        
    def get_wide(variable, start_date, end_date):
        
        
        
    