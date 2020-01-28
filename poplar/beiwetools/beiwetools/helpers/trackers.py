'''
Classes for doing online summaries during data processing tasks.
'''
import os
import logging
import datetime
import numpy as np
from collections import OrderedDict
from .process import to_1Darray
from .functions import write_json
from .time_constants import local_time_format


logger = logging.getLogger(__name__)


class SamplingSummary():
    '''
    Track intersample times from time series data that may be distributed over multiple locations.
    Used to summarize sampling frequency for Beiwe passive data streams.

    Args:
        cutoff (int): A maximum intersample duration in the time units of the data, e.g. milliseconds.
            Intersample times above this cutoff are ignored.
            Typically, this would be set to a value near the sensor's off-duration.
        start_with (numpy.ndarray): Optional.  
            Use this argument to intialize the counts array.
        
    Attributes:
        cutoff (int): Intersample times above this cutoff are ignored.
        counts (numpy.ndarray): A 1-D array of counts of intersample times.
            For example, counts[1] is the number of intersample times of length 1 unit.
        n (int): Total number of intersample times.  (Sum of counts.)
        track_last (bool): See SamplingSummary.update() for details.
        last (int): Last time from previous update. Used only if track_last is True.
    '''
    def __init__(self, cutoff, start_with = None, track_last = True):
        self.cutoff = cutoff
        if not start_with is None:
            if len(start_with) != cutoff+1:
                logging.warning('Array of starting counts is too long or too short.')
            self.counts = start_with
        else:
            self.counts = np.zeros(self.cutoff+1)
        self.n = np.sum(self.counts)
        self.track_last = track_last
        self.last = None
        
    def update(self, new, is_sorted = True):
        '''
        Update counts with information from additional time series data.
        
        Args:
            new (list, array, or DataFrame): 
                A list or 1-D array of sampling times (e.g. timestamps).
                Can also be a pandas dataframe with a "timestamp" column.
            is_sorted (bool): True if new has already been sorted in increasing order.
            track_last (bool):
                If True, the last time from the previous update is prepended to the next update.
                This accounts for the intersample time between updates.

        Returns:
            None
        '''        
        new = to_1Darray(new, 'timestamp')
        if len(new) > 0:
            if self.track_last:
                if not self.last is None:
                    t = np.insert(t, 0, self.last)             
                self.last = t[-1]
            if not is_sorted:
                t = np.sort(t, kind = 'heapsort')
            temp = np.diff(t)
            for i in temp:
                if i <= self.cutoff:  self.counts[i] += 1
            self.n = np.sum(self.counts)

    def frequency(self, per = 1000):
        '''
        Estimate the number of samples expected in a given duration.
        
        Args:
            per (int): A duration in the time units of the data.
                For example, use per = 1000 to get Hz from millisecond timestamps.
            
        Returns:
            f (float): Expected number of samples.
        '''        
        duration = np.sum(self.counts * np.arange(self.cutoff+1))
        if duration == 0:
            return(None)
        else:
            f = (self.n / duration) * 1000
            return(f)

    def save(self, directory, filename = 'intersample_times'):
        '''
        Save counts to a text file.
        
        Args:
            directory (str): Where to write the text file.
            filename (str): Name for the text file.
            
        Returns:
            None
        '''
        np.savetxt(os.path.join(directory, filename + '.txt'), self.counts)


class StatsTracker():
    '''
    Template for classes that do online calculation of statistics.

    Args:
        name (str): Name of the variable to track.
            Must be provided if doing updates with a pandas dataframe.
        labels (list): Names of summary statistics that are tracked.
        starting_values (list): How to initialize summary statistics.
        to_get (list): Names of summary statistics that are returned by get().
        to_save (list): Names of summaries to save to text file.
            
    Attributes:
        stats (OrderedDict): Dictionary of summary statistics.
        Other attributes: Same as args.
    '''
    def __init__(self, name = '', labels = [], starting_values = [], 
                 to_get = [], to_save = []):
        self.name = name
        self.labels = labels
        self.starting_values = starting_values
        self.to_get = to_get
        self.to_save = to_save
        self.stats = OrderedDict(zip(self.labels, self.starting_values))
        
    def update(self, new):
        '''
        How to update stats with a list, array, or dataframe of new observations.

        Args:
            new (list, array, or DataFrame): New observations.
            
        Returns:
            None
        '''
        new = to_1Darray(new, self.name)
        pass

    def get(self):
        '''
        Return some summary statistics.
        
        Args:
            None
            
        Returns:
            to_return (OrderedDict): Subset of stats.
        '''
        to_return = OrderedDict()
        for s in self.to_get:
            if len(self.name) > 0: k = self.name + '_' + s
            else: k = s
            to_return[k] = self.stats[s]
        return(to_return)
        
    def save(self, directory):
        '''
        Save some stats to a text file.

        Args:
            directory (str): Where to write the text file.
                
        Returns:
            None
        '''
        for s in self.to_save:
            if len(self.name) > 0: filename = self.name + '_' + s
            else: filename = s
            to_write = self.stats[s]
            if isinstance(to_write, (dict, OrderedDict)):
                write_json(to_write, filename, directory)
            elif isinstance(to_write, np.ndarray):
                np.savetxt(os.path.join(directory, filename + '.txt'), to_write)
            else: logger.warning('Unable to save this type.')
            

class HistogramTracker(StatsTracker):
    '''
    Given bins, track the histogram of a continuous variable.
    '''
    pass            


class CategoryTracker(StatsTracker):
    '''
    Track levels of a categorical variable.    
    '''
    def __init__(self, name = '', 
                 to_get = ['unique_categories'], to_save = ['counts']):    
        super().__init__(name, labels = ['unique_categories', 'counts'], 
                               starting_values = [0, OrderedDict({})],
                               to_get = to_get, to_save = to_save)
    
    def update(self, new):
        new = to_1Darray(new, self.name)
        if len(new) > 0:
            for n in new:
                k = str(n)
                if k in self.stats['counts'].keys():
                    self.stats['counts'][k] += 1                
                else:
                    self.stats['counts'][k] = 1  
                    self.stats['unique_categories'] += 1


class RangeTracker(StatsTracker):
    '''
    Track min/max values.
    '''
    def __init__(self, name = '', to_get = ['min', 'max'], to_save = []):    
        super().__init__(name, labels = ['min', 'max'], 
                               starting_values = [None, None],
                               to_get = to_get, to_save = to_save)
    
    def update(self, new):
        new = to_1Darray(new, self.name)    
        if len(new) > 0:
            temp_min = np.min(new)
            temp_max = np.max(new)
            if self.stats['min'] is None and self.stats['max'] is None:
                self.stats['min'] = temp_min
                self.stats['max'] = temp_max
            else:
                self.stats['min'] = np.min([self.stats['min'], temp_min])
                self.stats['max'] = np.max([self.stats['max'], temp_max])


class Welford(StatsTracker):
    '''
    Implementation of Welford's algorithm for online computation of variance.
    From: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Note:
        n (int): Sample size.
        mean (float): Sample mean.
        sos (float): Sum of squares.
        var, std (float): Sample variance and standard deviation.
    '''
    def __init__(self, name = '', to_get = ['n', 'mean', 'var'], to_save = []):
        super().__init__(name, labels = ['n', 'mean', 'sos', 'var', 'std'], 
                               starting_values = [0, 0, 0, None, None],
                               to_get = to_get, to_save = to_save)
    
    def update(self, new):
        new = to_1Darray(new, self.name)    
        if len(new) > 0:
            for obs in new: self.update_increment(obs) 

    def update_increment(self, obs):
        '''
        Update with a single new observation.
        '''
        self.stats['n'] += 1
        n = self.stats['n']
        delta = obs - self.stats['mean']
        self.stats['mean'] += delta / n
        delta2 = obs - self.stats['mean']
        self.stats['sos'] += delta * delta2
        if n > 1:
            self.stats['var'] = self.stats['sos']/(n-1)
            self.stats['std'] = np.sqrt(self.stats['var'])
