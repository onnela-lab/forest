'''
Templates for processing tasks.
'''
import inspect
import logging
from collections import OrderedDict
from .classes import Timer
from .trackers import trackers
from .functions import write_json


logger = logging.getLogger(__name__)


def setup_output(proc_dir, process_name, stream_name):
    '''
    Create files and folders for output.
    

    Args:
        proc_dir (str): Path to directory for processed data output.
        process_name (str): Name of the process, e.g. 'summary', 'calibration', etc.
        stream_name (str): Name of a Beiwe data stream, e.g. 'accelerometer', 'gps', etc.
        
    Returns:
        output_dict (dict): Dictionary of paths to files and directories for summary output.
    '''
    # set up directories
    stream_dir = os.path.join(proc_dir, stream_name)
    


    acc_dir = os.path.join(proc_dir, 'accelerometer')
    
    summary_dir = os.path.join(acc_dir, 'summary')
    dt_dir = os.path.join(summary_dir, local_now().replace(' ', '_'))
    records_dir = os.path.join(dt_dir, 'records')
    out_dir = os.path.join(dt_dir, 'users')
    setup_directories([acc_dir, summary_dir, dt_dir, records_dir, out_dir])
    # set up files    
    summary_records = setup_csv('summary_records', records_dir, 
                                header = summary_records_header)        
    # return paths
    output_dict = dict(zip(['records_dir', 'out_dir', 'summary_records'], 
                          [records_dir, out_dir, summary_records]))
    return(output_dict)


def f_of_kwargs(f, kwargs):
    '''
    This function gets f's keyword arguments from kwargs, and then 

    Args:
        
        
    Returns:
        

    '''    
    pass


def get_new_kwargs(functions, returns, kwargs):
    '''
    
    
    Returns:
        new_kwargs (dict): Dictionary of 

    '''    
    pass




#test_2 = returns_to_dict(test, ['x', 'y', 'z'])
#
#test_2(d)
#test_2('a' = 1, 'b' = 2, 'c' = 3)
#
#    
#d = {'a':1, 'b':2, 'c':3}
#
#test(**{'a':1, 'b':2, 'c':3})
#
#
#    
#def testf(): return(1, 2 ,3 )
#list(zip(['a', 'b', 'c'], testf()))
#
#
#    
#
#from beiwetools.helpers.functions import write_json
#
#import inspect
#inspect.getargspec(write_json).args # get argument names
#
#
#write_json.__code__.co_varnames
#
#s = {'a':1, 'b':2}
#t = {'b':1, 'c':3}
#s.update(t)
#t.update(s)
#


class ProcessTemplate():
    '''
    This class is a template for raw data processing tasks, e.g. summary, calibration, etc.
    The purpose of this class is to get do(), a function that processes lots of files.
    The input for do() is a list of users and dictionaries of keywords.
    
    The do() function then does the following:
    
        Set up process parameters   (1)
        for u in users:
            Set up user parameters  (2)
            Process user's files    (3)
            Update user records     (4)
        Update process records      (5)
    
    Each step corresponds to a list of functions.  
    - Step (1) probably includes a function to set up output files and folders.
    - Steps (2), (3), (4) could just be one list, but are split up for legibility.
    - Depending on the process, functions in Step (3) may implement one or more of the following:
        - A loop through a user's files,
        - A loop through sections of files (e.g. 60-second windows),
        - A loop through multiple files at a time (e.g. all files from a 5-day period).
    - The do() function manages kwargs while processing tasks. Therefore:
        - Keywords should be used consistently everywhere.
        - If a function returns something, it should be in the form of a dictionary of keyword arguments.  The beiwetools.helpers.decorators submodule provides some tools to simplify this.
    - See accrep.do_summary() for an example.

    Args: 
        setup_process   (list): List of functions to call at step (1).
        setup_user      (list): Functions to call at step (2).
        process_user    (list): Functions to call at step (3).
        user_records    (list): Functions to call at step (4).
        process_records (list): Functions to call at step (5).                                    

    Attributes:
        Same as args.
        '''
    def __init__(self, setup_process, setup_user, process_user,
                 user_records, process_records):
        
        pass
        
    def do(self, user_ids, user_dictionaries, kwargs, verbose = True):
        '''
        Args:
            user_ids (list): List of Beiwe user IDs with data to process.
            process_kwargs (dict): Dictionary of keyword arguments.
            
        Returns:
        '''
        # save a record of kwargs
        pass
        
