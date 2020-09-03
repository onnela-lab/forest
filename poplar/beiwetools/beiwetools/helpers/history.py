'''Classes for tracking changes in categorical variables over time.

'''
from logging import getLogger
from collections import OrderedDict
from beiwetools.helpers.time_constants import time_unit_ms


logger = getLogger(__name__)


class HistoryRecord():
    '''
    Class for tracking changes in a single categorical variable.
    Use a HistoryRecord to track changes in timezones, device parameters, etc.
                
    Attributes:
        variable_name (str or Nonetype): Optional name for the variable that
            is tracked by the class instance.
        levels (list): Sorted list of variable levels that appear in the 
            history.
        transition_dictionary (OrderedDict): Each key is a timestamp, each
            value is a level of the categorical variable.  Timestamps appear
            chronologically, with no timestamp appearing more than once.
            Each timestamp corresponds to a time at which the value of the 
            variable changed.
        
    '''
    def __init__(self):
        '''
        Initialize an empty object.
        '''
        pass
        
    @classmethod
    def create(cls, variable_name, transition_dictionary):
        '''
        Preferred method for creating 
        '''
        self = cls.__new__(cls)
        if isinstance(transition_dictionary, dict):
            transition_dictionary = OrderedDict(transition_dictionary)
        # check dictionary        

        #
        self.transition_dictionary = transition_dictionary
        self.variable_name = variable_name
        self.levels = sorted(list(set(transition_dictionary.values())))
        return(self)

    def export(cls, out_dir, filename = None):
        pass
        
    @classmethod
    def load(cls, history_dir):
        pass

    def update(cls, new_transition_dictionary):
        pass
    
    def get_epochs(self):
        '''
        Identify epochs, time ranges when the variable does not change.                                         
        '''
        pass

    def get_range_values(self):
        '''
        Get values 
        '''
    
    def get_value(self):
        pass
        
    def get_details(self):
        pass


    def print_dict(self):
        print(self.__dict__)

    def change_locals(self):
        pass

test3 = TimestampRange()
test3.__dict__

test == test2

test2 = HistoryRecord()
test2.print_dict()

test.__dict__
test2.__dict__ = test.__dict__

test.transition_dictionary
test2.transition_dictionary

test = HistoryRecord.create('apple', dict({('a', 1)}))

test.print_dict()
test.__dict__['variable_name'] = 'qwer'
test.variable_name

test1 = TimestampRange.create(1, 2)
test2 = TimestampRange.create(1, 2)
test1 == test2

test1.__dict__ == test2.__dict__
    

class TimestampRange():
    '''
    Class for representing a period of time between two timestamps.

    Attributes:
        start_timestamp (int): First timestamp in the time period.
        end_timestamp (int):   Last timestamp in the time period.
    '''
    @classmethod
    def create(cls, start_timestamp, end_timestamp):
        '''
        Create a new TimestampRange.
        '''
        self = cls.__new__(cls)
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp        
        return(self)

    def get(self):
        '''
        Get start/end timestamps as a tuple.
        '''
        range_tuple = (self.start_timestamp, self.end_timestamp)
        return(range_tuple)

    def duration(self, time_unit):
        '''
        Get the duration of the time period.

        Args:
            time_unit (str): a key from TIME_MS, e.g. 'minutes' or 'days.'
            
        Returns: 
            period_duration (float): The length of the time period in the 
                corresponding units.        
        '''
        unit_ms = time_unit_ms[time_unit]
        period_duration = (self.end_timestamp - self.start_timestamp) / unit_ms
        return(period_duration)

    def contains(self, x):
        '''
        Check if a timestamp or another time period is contained within 
        the range.

        Args:
            x (int or TimestampRange): Either a single timestamp (int) or 
                a representation of a time period (TimestampRange).                
            
        Returns:
            (bool):  True if x occured during the time period.
        '''
        if isinstance(x, TimestampRange):
            start_check = self.start_timestamp <= x.start_timestamp
            end_check   = self.end_timestamp >= x.end_timestamp
            return(start_check and end_check)            
        elif isinstance(x, int):
            return(x >= self.start_timestamp and x <= self.end_timestamp)            
        else: logger.warning('Incompatible type: %s' % type(x))
        
        
        