''' Classes for working with raw Fitabase data.
'''
import os
import logging
from collections import OrderedDict
from beiwetools.helpers.time import local_now
from .functions import parse_filename


logger = logging.getLogger(__name__)


class FitabaseRegistry():
    '''
    Represent available data in a directory of Fitabase data sets.
    
    Attributes:
        directory (str): Path to raw Fitabase files.
        ids (list): Sorted list of Fitabase ids with available data.
        types (list): Sorted list of available file types.
        lookup (OrderedDict): Keys are Fitabase ids, values are OrderedDicts.        
            path[<fitabase_id>][<file_type>] is the name of the corresponding file.    
        flagged_files (list): List of file names with duplicate file types.
            These file names are not included in the registry.
    '''
    @classmethod
    def create(cls, fitabase_dir):
        '''
        Get registry information for a directory of Fitabase data.

        Args:
            fitabse_dir (str): Path to a directory of Fitabase data.
            
        Returns:
            self (FitabaseDirectory)
        
        Note:
            All files in a directory should have the same start/end dates.
            Each fitabase identifier should have only one file per file type.
        '''
        self = cls.__new__(cls)  
        self.directory = fitabase_dir
        file_names = sorted(os.listdir(self.directory))
        lookup = OrderedDict()
        fitabase_ids = []        
        file_types = []
        flagged_files = []
        start = []
        end = []
        for f in file_names:
            p = parse_filename.to_dict(f)
            fid = p['fitabase_id']
            ft  = p['file_type']
            file_types.append(ft)
            start.append(p['local_start'])
            end.  append(p['local_end'])
            if not fid in lookup:
                lookup[fid] = OrderedDict()
            if not ft in lookup[fid]:
                lookup[fid][ft] = f
            else:
                logger.warning('Multiple %s files for identifier %s.' %(ft, fid))
                flagged_files.append(f)
        if len(set(start)) > 1:  logger.warning('Multiple start dates.')
        if len(set(end))   > 1:  logger.warning('Multiple end dates.')
        self.ids = sorted(list(set(fitabase_ids)))
        self.types = sorted(list(set(file_types)))
        self.lookup = lookup
        self.flagged_files = flagged_files
        logging.info('Created registry for %d Fitabase identifiers and %d file types.' % (len(self.ids), len(self.types)))
        return(self)
        
    @classmethod
    def load(cls, directory):
        '''
        Load an exported FitabaseDirectory from json files.
        '''
        pass

    def export(self, name, directory, track_time = True):
        '''
        Save json files with fitabase directory records.  
        Overwrites pre-existing records.
        
        Args:
            name (str): Save files to a folder with this name.
            directory (str): Where to save folder of records.
            track_time (bool): If True, export to a timestamped sub-directory.
            
        Returns:
            directory (str): Path to exported records.
        '''   
        directory = os.path.join(directory, name)
        if track_time:
            temp = 'Fitabase registry export from ' + local_now()
            directory = os.path.join(directory, temp.replace(' ', '_'))
        # export(self, directory)
        # return(directory)
        pass
        