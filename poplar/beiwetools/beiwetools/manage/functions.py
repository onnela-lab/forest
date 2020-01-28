'''Helpers for beiwetools.manage.classes.

'''
import os
import logging
import pandas as pd

from humanize import naturalsize
from collections import OrderedDict

from beiwetools.helpers.time import summarize_UTC_range
from beiwetools.helpers.functions import (sort_by, setup_directories, 
                                          write_json, read_json, 
                                          setup_csv, write_to_csv)

from .headers import info_header


logger = logging.getLogger(__name__)


# get names for passive data streams and survey directories
this_dir = os.path.dirname(__file__)
try:
    data_streams = read_json(os.path.join(this_dir, 'data_streams.json'))
    passive_data = data_streams['passive_data'] # raw passive data directories
    survey_data = data_streams['survey_data']   # raw survey data directories
    # drop 'identifiers' and sort passive data streams:
    passive_available = {'both': [], 'iOS': [], 'Android': []}
    for k in passive_data.keys():
        if k != 'identifiers':
            passive_available['both'].append(k)
            if passive_data[k]['iOS']: passive_available['iOS'].append(k)                                
            if passive_data[k]['Android']: passive_available['Android'].append(k)
            if not passive_data[k]['Android'] and not passive_data[k]['iOS']:    
                logger.warning('Check data stream records for \'%s\' .' %k)
except:
    logger.warning('There\'s a problem with the data stream records.')


def check_dirs(dirs):
    '''
    Args:
        dirs (list):  List of paths to directories.    

    Returns:
        not_exist (list): Paths in dirs that don't exist.        
        not_dir (list):   Paths in dirs that exist but aren't directories.
        empty (list):     Paths in dirs that exist but are empty.
        not_empty (list): Paths in dirs that contain files or folders.
    '''
    not_exist, not_dir, empty, not_empty = [], [], [], []
    for d in dirs:
        if not os.path.exists(d):
            not_exist.append(d)
        else:
            if not os.path.isdir(d):
                not_dir.append(d)
            else:
                if len(os.listdir(d)) == 0:
                    empty.append(d)
                else:
                    not_empty.append(d)
    return(not_exist, not_dir, empty, not_empty)    
    

def merge_contents(data_dirs, UTC_range = None):
    '''
    Helper function for passive_registry and survey_regsitry.
    Discards paths to duplicate files and chooses larger files whenever possible.
    
    Args:    
        data_dirs (list):  
            List of paths to directories that may contain files with duplicate names.
        UTC_range (list or Nonetype): Optional.  
            Ordered pair of date/times in filename_time_format, [start, end].
            If not None, ignore files before start and after end.
            
    Returns:
        merge (list):  
            A list of paths in which no basename is duplicated, sorted in order of basenames.
    '''    
    if isinstance(data_dirs, str): data_dirs = [data_dirs]
    if len(data_dirs) == 1:
         d = data_dirs[0]
         file_names = sorted(os.listdir(d))
         if not UTC_range is None:
             start, end = [dt + '.csv' for dt in UTC_range]
             file_names = [n for n in file_names if n >= start and n <= end]
         merge = [os.path.join(d, f) for f in file_names]        
    else:
        file_dictionary = OrderedDict()
        for d in data_dirs:
            for f in os.listdir(d):
                if f in file_dictionary.keys():
                    file_dictionary[f].append(d)
                else:
                    file_dictionary[f] = [d]
        if not UTC_range is None:
            start, end = [dt + '.csv' for dt in UTC_range]
            file_names = list(file_dictionary.keys())
            for f in file_names:
                if f < start or f > end: del file_dictionary[f]
        for f in file_dictionary.keys():        
            file_paths = [os.path.join(d, f) for d in file_dictionary[f]]
            file_sizes = [os.path.getsize(p) for p in file_paths]
            file_dictionary[f] = file_paths[file_sizes.index(max(file_sizes))]
        # sort values by keys
        merge = sort_by(list(file_dictionary.values()), list(file_dictionary.keys()))
    return(merge)


def get_survey_ids(dirs):
    '''
    Args:
        dirs (list): List of paths to directories that contain a survey type.
    
    Returns:
        sids (list): 
            List of survey ID folders that are found in at least one of dirs.    
        files (list):
            Paths to raw files, which may be found in some older survey type folders.
    '''
    sids, files = [], []
    for d in dirs:
        try: temp = os.listdir(d)        
        except: temp = []
        sids += [t for t in temp if os.path.isdir (os.path.join(d, t))]
        raw   = [f for f in temp if os.path.isfile(os.path.join(d, f))]
        files += [os.path.join(d, f) for f in raw]
    sids =  sorted(list(set(sids)))
    files.sort()
    return(sids, files)


def size_on_disk(filepaths, ndigits = 1):
    '''
    Given a list of filepaths, return total size on disk in megabytes.
    
    Args: 
        filepaths (list): List of paths to files.        
        ndigits (int or Nonetype):
            If not None, number of digits for rounding.
    Returns:
        b (int): Total size of all files, in bytes. 
    '''
    b = 0
    if isinstance(filepaths, str): filepaths = [filepaths]
    for p in filepaths:
        b += os.path.getsize(p)
    return(b)


def identifiers_registry(user_id, raw_dirs, UTC_range = None):
    '''
    Get registry of identifiers for one user.
       
    Args:
        user_id (str): Beiwe user ID.
        raw_dirs (list): 
        raw_dirs (str or list):  
            Paths to directories that may contain raw data from this user.
        UTC_range (list or Nonetype):  
            Ordered pair of date/times in filename_time_format, [start, end].
            If not None, ignore files before start and after end.

    Returns:    
        registry (OrderedDict): Keys and values are:
            'flag':  Values are either:
                'not found' - Identifiers folder is missing or empty.
                None - Identifiers folder exists and is not empty.
            'count': Number of identifiers files (int).
            'bytes':    Total size of files on disk in bytes.                
            'files': List of all available identiers files.
    '''
    registry = OrderedDict({'files': [], 'flag': None, 'count': 0, 'bytes': 0})
    to_check = [os.path.join(d, user_id, 'identifiers') for d in raw_dirs]
    not_exist, not_dir, empty, not_empty = check_dirs(to_check)        
    if len(not_empty) == 0: 
        registry['flag'] = 'not found'
        logger.warning('No identifiers found for %s.' % user_id)        
    else: 
        merge = merge_contents(not_empty, UTC_range)
        if len(merge) == 0:                
            logger.warning('No identifiers found for %s in this range.' % user_id)        
            merge = [merge_contents(not_empty, ['1970-01-01 00_00_00', UTC_range[1]])[-1]]
            if len(merge) > 0:
                logger.warning('Using last observed identifiers file for %s.' % user_id)
            else:
                registry['flag'] = 'not found'
                logger.warning('No identifiers found for %s.' % user_id)        
    registry['files'] = merge
    registry['count'] = len(merge)
    registry['bytes'] = size_on_disk(merge)
    return(registry)
    
    
def passive_registry(user_id, phone_os, raw_dirs, UTC_range = None):
    '''
    Get registry of raw passive data for one user.
       
    Args:
        user_id (str): Beiwe user ID.
        os (str): 'iOS' or 'Android' or 'both'.
        raw_dirs (list): 
        raw_dirs (str or list):  
            Paths to directories that may contain raw data from this user.
        UTC_range (list or Nonetype):  
            Ordered pair of date/times in filename_time_format, [start, end].
            If not None, ignore files before start and after end.

    Returns:    
        passive_range (list): 
            Datetimes corresponding to first and last passive data files.
        registry (OrderedDict):  Keys are passive data streams.
            Each value is an ordered dictionary with keys and values:
                'flag':  Values may be:
                    'not available for OS' - Data stream doesn't exist for this device type.
                    'not found' - Data stream directory is missing or empty.
                    None - Data stream exists for the device and is not empty.
                'count': Number of files for this data stream (int).
                'bytes':    Total size of files on disk in bytes.                
                'files': List of all available files for the data stream.
    '''
    passive_range = []
    registry = OrderedDict.fromkeys(passive_available['both'])
    for stream in passive_available['both']:
        temp = OrderedDict({'flag': None, 'count': 0, 'bytes': 0, 'files': []})
        if not stream in passive_available[phone_os]:
            temp['flag'] = 'not available for OS'
        else:
            to_check = [os.path.join(d, user_id, stream) for d in raw_dirs]
            not_exist, not_dir, empty, not_empty = check_dirs(to_check)          
            if len(not_empty) == 0: temp['flag'] = 'not found'
            else: 
                merge = merge_contents(not_empty, UTC_range)
                passive_range += [os.path.basename(merge[0]).split('.')[0], 
                                  os.path.basename(merge[-1]).split('.')[0]]
                temp['files'] = merge
                temp['count'] = len(merge)
                temp['bytes'] = size_on_disk(merge)
        registry[stream] = temp
    passive_range.sort()
    if len(passive_range) > 1:
        passive_range = [passive_range[0], passive_range[-1]]
    return(passive_range, registry)


def survey_registry(user_id, raw_dirs, UTC_range = None):
    '''
    Get registry of survey data for one user.
       
    Args:
        user_id (str): Beiwe user ID.
        raw_dirs (list): 
        raw_dirs (str or list):  
            Paths to directories that may contain raw data from this user.
        UTC_range (list or Nonetype):  
            Ordered pair of date/times in filename_time_format, [start, end].
            If not None, ignore files before start and after end.

    Returns:
        survey_range (list): 
            Datetimes corresponding to first and last survey data files.
        registry(OrderedDict):  
            Keys are names of survey directories (e.g. 'audio_recordings', 'survey_timings').
            Each value is an ordered dictionary with keys and values:
                'flag':  None or 'not found'.
                'ids': 
                    An ordered dictionary. Keys are survey identifiers.  
                    Each value is an ordered dictionary with keys and values:
                        'count': Number of files for this data stream (int).
                        'bytes':    Total size of files on disk in bytes.                
                        'files': List of all available files for the corresponding survey.
        not_registered (list): 
            Paths to unregistered files in irregular directories.
            Irregular directories are survey directories that contain raw data files.
    '''
    survey_range = []
    not_registered = []
    registry = OrderedDict.fromkeys(survey_data)
    for survey_type in survey_data:
        registry[survey_type] = OrderedDict({'flag': None, 'ids': OrderedDict()})
        survey_dirs = [os.path.join(d, user_id, survey_type) for d in raw_dirs]
        sids, files = get_survey_ids(survey_dirs)
        not_registered += files
        if len(sids) == 0:
            registry[survey_type]['flag'] = 'not found'
        else:
            for s in sids:
                temp = OrderedDict({'count': 0, 'bytes': 0, 'files': []})
                to_check = [os.path.join(d, s) for d in survey_dirs]                        
                not_exist, not_dir, empty, not_empty = check_dirs(to_check)          
                if len(not_empty) > 0: 
                    merge = merge_contents(not_empty, UTC_range)        
                    survey_range += [os.path.basename(merge[0]).split('.')[0], 
                                     os.path.basename(merge[-1]).split('.')[0]]
                    temp['files'] = merge
                    temp['count'] = len(merge)
                    temp['bytes'] = size_on_disk(merge)
                registry[survey_type]['ids'][s] = temp
    if len(survey_range) > 1:
        survey_range.sort()
        survey_range = [survey_range[0], survey_range[-1]]
    return(survey_range, registry, not_registered)


def registry_to_text(passive, surveys, first, last, names):
    '''
    Generate a text summary of a passive data and survey registries.
    '''    
    # passive
    r, hours, u = summarize_UTC_range([first, last], unit = 'hours', ndigits = None)
    streams = [s for s in passive if passive[s]['flag'] is None]
    p = pd.DataFrame(columns = ['  Files', '  Coverage', '    Storage'])
    for s in streams:
        if s in ['identifiers', 'calls', 'texts', 'proximity']:
            p.loc[s.ljust(15)] = [passive[s]['count'], 
                        None,
                        naturalsize(passive[s]['bytes'])]        
        else:
            p.loc[s.ljust(15)] = [passive[s]['count'], 
                        round(passive[s]['count']/hours, 2),
                        naturalsize(passive[s]['bytes'])]
    if len(p) > 0:
        p_text = '\n' + p.to_string(na_rep = '-')
    else: p_text = None
    # surveys
    s_types = [t for t in surveys if surveys[t]['flag'] is None]
    s_text = []
    for st in s_types:
        temp = surveys[st]['ids']
        s = pd.DataFrame(columns = ['  Files', '    Storage'])
        for sid in temp:
            try: name = names[sid]
            except: name = sid
            if len(name) > 25: index = name[0:22] + '...'
            else: index = name.ljust(25)
            s.loc[index] = [temp[sid]['count'], naturalsize(temp[sid]['bytes'])]
        s.sort_index(inplace = True)
        s_text.append(s.to_string(na_rep = '-'))
    return(p_text, ['\n' + st for st in s_types], s_text)

def data_to_text(passive, surveys, data, object_names):
    '''
    Get a text summary of data streams for multiple users.
    '''    
    # passive streams
    p = pd.DataFrame(columns = ['  Files', '    Storage'])
    for s in passive:
        count, size = 0, 0
        for d in data.values():
            if s in d.passive:
                count += d.passive[s]['count']
                size += d.passive[s]['bytes']
        p.loc[s.ljust(15)] = [count, naturalsize(size)]        
    if len(p) > 0:
        p_text = '\n' + p.to_string(na_rep = '-')
    else: p_text = None
    # survey data
    # surveys
    s_text = []
    for st in surveys:
        temp = surveys[st]
        s = pd.DataFrame(columns = ['  Files', '    Storage'])
        for sid in temp:
            try: name = object_names[sid]
            except: name = sid
            if len(name) > 25: index = name[0:22] + '...'
            else: index = name.ljust(25)
            count, size = 0, 0
            for d in data.values():
                if sid in d.surveys[st]['ids']:
                    count += d.surveys[st]['ids'][sid]['count']
                    size += d.surveys[st]['ids'][sid]['bytes']
            s.loc[index] = [count, naturalsize(size)]
        s.sort_index(inplace = True)
        s_text.append(s.to_string(na_rep = '-'))
    return(p_text, ['\n' + st for st in list(surveys.keys())], s_text)


def export_manage(d, directory):
    '''
    Handle exports for beiwetools.manage.classes.
    '''
    # if isinstance(d, DeviceInfo):
    if str(type(d)) == "<class 'beiwetools.manage.classes.DeviceInfo'>": 
        try:
            temp = list(d.identifiers.values())[-1]
            filename = temp['patient_id'] + '_identifiers'
            header = ['from_file'] + list(temp.keys())
            path = setup_csv(filename, directory, header)
            for f in d.identifiers.keys():  
                line = [f] + list(d.identifiers[f].values())
                write_to_csv(path, line)
        except:
            logger.warning('Unable to export identifiers.')
    # elif isinstance(d, UserData):
    elif str(type(d)) == "<class 'beiwetools.manage.classes.UserData'>": 
        out = OrderedDict([('id',        d.id),
                           ('passive',   d.passive),
                           ('surveys',   d.surveys),
                           ('first',     d.first),
                           ('last',      d.last),
                           ('UTC_range', d.UTC_range),
                           ('not_registered', d.not_registered)])
        write_json(out, d.id + '_registry', directory)    
    # elif isinstance(d, BeiweProject):
    elif str(type(d)) == "<class 'beiwetools.manage.classes.BeiweProject'>": 
        folder_names = ['identifiers', 'user_summaries', 'records']
        dirs = [os.path.join(directory, f) for f in folder_names]
        dirs.append(os.path.join(directory, 'records', 'registries'))
        setup_directories(dirs)        
        idp, usp, recp, regp = dirs
        csvp = setup_csv('overview', directory, info_header + ['study_name', 'configuration_files'])
        for i in d.ids:
            ud = d.data[i]
            ud.export(regp)
            ud.device.export(idp)
            ud.summary.to_file(i + '_summary', usp)
            try: extra_info = [d.lookup['study_name'][i], len(d.lookup['configuration'][i])]
            except: extra_info = [None, None]
            write_to_csv(csvp, list(ud.info.values()) + extra_info)
        out = OrderedDict([('ids', d.ids),
                           ('raw_dirs', d.raw_dirs),
                           ('first', d.first),
                           ('last', d.last),
                           ('passive', d.passive),
                           ('surveys', d.surveys),
                           ('lists', d.lists),
                           ('lookup', d.lookup),
                           ('flags', d.flags)])
        write_json(out, 'export', recp)
        d.summary.to_file('summary', directory)
    else:        
        logger.warning('This function doesn\'t handle export of %s.' % str(type(d)))


def load_manage(d, path):        
    '''
    Handle loading for beiwetools.manage.classes.
    '''
    # if isinstance(d, UserData):    
    if str(type(d)) == "<class 'beiwetools.manage.classes.UserData'>": 
        temp = read_json(path)
        d.id        = temp['id']
        d.passive   = temp['passive']
        d.surveys   = temp['surveys']
        d.first     = temp['first']
        d.last      = temp['last']  
        d.UTC_range = temp['UTC_range']
        d.not_registered = temp['not_registered']
    # elif isinstance(d, BeiweProject):        
    elif str(type(d)) == "<class 'beiwetools.manage.classes.BeiweProject'>": 
        export = os.path.join(path, 'records', 'export.json')
        temp = read_json(export)
        d.ids = temp['ids']
        d.raw_dirs = temp['raw_dirs']
        d.first = temp['first']
        d.last = temp['last']
        d.passive = temp['passive']
        d.surveys = temp['surveys']
        d.lists = temp['lists']
        d.lookup = temp['lookup']
        d.flags = temp['flags']
    else:
        logger.warning('This function doesn\'t handle loading for %s.' % str(type(d)))