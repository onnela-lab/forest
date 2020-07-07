'''
Templates for processing tasks.
'''
import logging


logger = logging.getLogger(__name__)


class ProcessTemplate():
    '''
    This class is a template for raw data processing tasks, e.g. summary, calibration, etc.
    The purpose of this class is to get the do() method, a function that processes lots of files.
    The inputs for do() might be: 
        - A list of users, 
        - Dictionaries of keyword arguments,
        - Instances of classes such as BeiweConfig, BeiweProject, etc.
    
    The do() function then does the following:
    
        (1) Set up process parameters
        (2) for u in list of users:
              (2a) Set up user parameters
              (2b) Process u's files
              (2c) Update u's records
        (3) Update process records
    
    Each step above corresponds to a list of functions.  
    - Step (1) probably includes a function to set up output files and folders.
    - Steps (2a), (2b), (2c) could be captured in a single task, but are split up for legibility.
    - Functions in Step (2b) may implement one or more of the following:
        - A loop through a user's files,
        - A loop through sections of files (e.g. 60-second windows),
        - A loop through multiple files at a time (e.g. all files from a 5-day period).
    - Functions in (2c) probably write output to files or folders created in step (1) and/or (2a).
	- Functions in (3) probably write output to files or folders created in step (1).

    The do() function manages kwargs while processing tasks. Therefore:
        - Keywords should be used consistently everywhere.
        - If a function returns something, it should be in the form of a dictionary of keyword arguments.  
        (The beiwetools.helpers.decorators module provides some tools to simplify this.)

    See accrep.do_summary() for an example.

    Attributes: 
        name (str): Name or description of process.
        setup_process   (list): List of functions to call at step (1).
        setup_user      (list): Functions to call at step (2a).
        process_user    (list): Functions to call at step (2b).
        user_records    (list): Functions to call at step (2c).
        process_records (list): Functions to call at step (3).                                    
    '''
    @classmethod
    def create(cls, name, setup_process, 
               setup_user, process_user,
               user_records, process_records):   

        self = cls.__new__(cls)        
        self.name = name
        self.setup_process = setup_process
        self.setup_user = setup_user
        self.process_user = process_user
        self.user_records = user_records
        self.process_records = process_records
        return(self)
        
    def do(self, user_ids, process_kwargs, id_lookup = {}):
        '''
        Args:
            user_ids (list): List of user IDs with data to process.
            process_kwargs (dict): Dictionary of keyword arguments.
            id_lookup (dict): 
                Use this dictionary to specify other identifiers.
                Keys are user ids (str).
                Values are alternate identifiers (str).
            
        Returns:
            None
        '''
        n = len(user_ids)
        # set up process parameters
        logger.info('%s: Setting up...' % self.name)
        for f in self.setup_process:
            process_kwargs.update(f.easy(process_kwargs))
        # loop through list of users
        for i in range(n):
            # set up user parameters
            logger.info('%s: Setting up for user %s of %s...' % (self.name, i, n))           
            uid = user_ids[i]
            if uid in id_lookup: uid = id_lookup[uid]
            user_kwargs = {'user_id': uid}
            user_kwargs.update(process_kwargs)
            for f in self.setup_user:
                user_kwargs.update(f.easy(user_kwargs))
            # process user data
            logger.info('%s: Processing data for user %s of %s...' % (self.name, i, n))                                   
            for f in self.process_user:
                user_kwargs.update(f.easy(user_kwargs))                
            # update user records
            logger.info('%s: Writing records for user %s of %s...' % (self.name, i, n))                                   
            for f in self.user_records:
                user_kwargs.update(f.easy(user_kwargs))                                
        # update process records
        logger.info('%s: Writing process records...' %self.name)                                   
        for f in self.process_records:
            process_kwargs.update(f.easy(self.kwargs))
        logger.info('%s: Finished.' %self.name)                                   


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
    
    
    # stream_dir = os.path.join(proc_dir, stream_name)
    


    # acc_dir = os.path.join(proc_dir, 'accelerometer')
    
    # summary_dir = os.path.join(acc_dir, 'summary')
    # dt_dir = os.path.join(summary_dir, local_now().replace(' ', '_'))
    # records_dir = os.path.join(dt_dir, 'records')
    # out_dir = os.path.join(dt_dir, 'users')
    # setup_directories([acc_dir, summary_dir, dt_dir, records_dir, out_dir])
    # # set up files    
    # summary_records = setup_csv('summary_records', records_dir, 
    #                            header = summary_records_header)        
    # # return paths
    # output_dict = dict(zip(['records_dir', 'out_dir', 'summary_records'], 
    #                      [records_dir, out_dir, summary_records]))
    # return(output_dict)
    pass


def setup_trackers():
	pass





        
