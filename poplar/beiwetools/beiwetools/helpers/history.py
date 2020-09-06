'''Classes for tracking changes in categorical variables over time.

'''
from logging import getLogger
from collections import OrderedDict
from beiwetools.helpers.time_constants import time_unit_ms
from .class_templates import SerializableClass


logger = getLogger(__name__)


class HistoryRecord(SerializableClass):
    '''
    Class for tracking changes in a single categorical variable.
    Use a HistoryRecord to track changes in timezones, device parameters, etc.
                
    Attributes:
        variable_name (str or Nonetype): Optional name for the variable that
            is tracked by the class instance.
        levels (list): Sorted list of variable levels that appear in the 
            history.
        transition_dictionary (OrderedDict): 
            - Each key is a timestamp.
            - Each value is a level of the categorical variable.  
            - Keys are chronological, with no timestamp appearing more 
            than once.
            - Each timestamp corresponds to a time at which the value of the 
            variable changed to a different level.        
    '''
    @classmethod
    def create(cls, variable_name, transition_dictionary):
        '''
        Create a new instance from a dictionary of transitions.
        '''
        self = cls.__new__(cls)
        if isinstance(transition_dictionary, dict):
            transition_dictionary = OrderedDict(transition_dictionary)
        if check_transitions(transition_dictionary):
            self.transition_dictionary = transition_dictionary
            self.variable_name = variable_name
            self.levels = sorted(list(set(transition_dictionary.values())))
            return(self)
        else:
            logger.warning('This is not a well-formed transition dictionary.')

    def update(self, new_transition_dictionary):
        '''
        Update transitions record with new information.
        '''
        if isinstance(new_transition_dictionary, dict):
            new_transition_dictionary = OrderedDict(new_transition_dictionary)
        if check_transitions(new_transition_dictionary):
            self.transition_dictionary.update(new_transition_dictionary)
            clean_transitions(self.transition_dictionary)
        else:
            logger.warning('This is not a well-formed transition dictionary.')            
    
    def get_epochs(self):
        '''
        Return epochs, time ranges when the variable does not change.                                         
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


def clean_transitions(dictionary):
    '''
    Remove non-transition events from a representation of transition times.
    An in-place operation; doesn't return anything.
    '''
    keys = list(dictionary.keys())
    to_remove = []
    for i in range(len(keys)-1):
        this_key = keys[i]
        next_key = keys[i+1]
        if dictionary[this_key] == dictionary[next_key]:
            to_remove.append(next_key)
    for k in to_remove: del dictionary[k]


def check_transitions(dictionary):
    '''
    Check if a dictionary is a valid representation of transition times.
    In a valid transition dictionary, keys are timestamps of events.
    Events are transitions from an old level to a new level of a variable.
    Each values is the new level observed at the corresponding time.    
        - Each key is a timestamp (int).
        - Each value is a level of the categorical variable.  
        - Keys are chronological, with no timestamp appearing more than once.
        - Each timestamp corresponds to a time at which the value of the 
          variable changed to a different level.        
    '''
    check = True
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    # integer keys
    key_check = [isinstance(k, int) for k in keys]
    if not all(key_check):
        check = False
        logger.warning('Keys must all be integers.')
    else:
        # strictly increasing keys    
        sort_check = [(y-x)>0 for x,y in zip(keys[:-1], keys[1:])]
        if not all(sort_check):
            check = False
            logger.warning('Keys must be in increasing order.')
        else:    
            # consecutive values are all different
            event_check = [x!=y for x,y in zip(values[:-1], values[1:])]
            if not all(event_check):
                check = False
                logger.warning('Values must be transition events.')
    return(check)    


class TimestampRange():
    '''
    Class for representing a period of time between two timestamps.
    To handle open-ended time ranges, use:
        - start_timestamp = 0
          (Unix epoch, 1970/01/01 00:00:00 UTC)
        or
        - end_timestamp = 2147483647000 
          (Last 32-bit Unix time, 2038/01/19 03:14:07 UTC)

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
            x (int or tuple or TimestampRange): 
                Either a single timestamp (int), 
                Or a tuple (start_timestamp, end_timestamp), 
                Or a representation of a time period (TimestampRange).                
            
        Returns:
            (bool):  True if x occured during the time period.
        '''
        if isinstance(x, tuple):
            try:
                x = TimestampRange.create(x[0], x[1])
            except:
                logger.warning('Incompatible tuple.')
        if isinstance(x, TimestampRange):
            start_check = self.start_timestamp <= x.start_timestamp
            end_check   = self.end_timestamp >= x.end_timestamp
            return(start_check and end_check)            
        elif isinstance(x, int):
            return(x >= self.start_timestamp and x <= self.end_timestamp)            
        else: logger.warning('Incompatible type: %s' % type(x))
        
    def overlaps(self, timestamp_range):
        '''
        Check if another time period overlaps this range.

        Args:
            
        Returns:
            (bool): True if 

        '''
        if isinstance(x, tuple):
            try:
                x = TimestampRange.create(x[0], x[1])
            except:
                logger.warning('Incompatible tuple.')
