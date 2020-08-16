'''Classes for working with directories of raw Beiwe data.

See examples/mange_example.ipynb for sample usage.
'''
import os
import logging

from humanize import naturalsize
from collections import OrderedDict

from beiwetools.helpers.time import summarize_UTC_range, local_now
from beiwetools.helpers.classes import Summary
from beiwetools.helpers.functions import check_same, sort_by, join_lists, coerce_to_dict
from beiwetools.configread.classes import BeiweConfig

from .headers import identifiers_header, info_header
from .functions import *


logger = logging.getLogger(__name__)


class BeiweProject():
    '''
    Class for organizing directories of raw Beiwe data from multiple users.

    Attributes:
        ids (list):  List of all available user IDs, sorted by first observation.
        raw_dirs (list): Paths to directories where raw data are found.
        data (OrderedDict):  Keys are Beiwe user ids, values are UserData objects.
        configurations (OrderedDict): 
            Keys are paths to configuration files.
            Values are corresponding BeiweConfig objects.
        first, last (str):  
            Date/time of first and last observations across all users.
            Formatted as '%Y-%m-%d %H_%M_%S'.
        passive (list):        
            List of available passive data streams across all users.
        surveys (OrderedDict):
            Keys are raw survey data types (e.g. 'audio_recordings', 'survey_answers').
            Values are lists of corresponding survey identifiers.
        lists (OrderedDict):
            Useful lists.  
            May be saved to JSON format, therefore should only contain primitive types.
            By default, includes:            
                'iOS': List of all users who only use iPhones.
                'Android': List of all users who only use Android phones.
        lookup (OrderedDict): 
            Dictionaries used for looking up user attributes and object names.
            May be saved to JSON format, therefore should only contain primitive types. 
            Values should be dictionaries in which keys are user IDs or object identifiers.
            Created with user lookups: 'study_name', 'default_name', 'os', 'configuration', 'UTC_range'
            If configuration files with unique names are attached:
                Also created with an object name lookup: 'object_name',
                And a reverse object name lookup: 'reverse_object_name'.
            Note:
                'study_name' and 'object_name' are assembled from configuration name assignments.
                If there are conflicts, later configuration paths take precedence.
        flags (OrderedDict): Values are lists of flagged identifiers.  Keys are:
            'ignored_users': User IDs in raw_dirs who are not included.
            'no_registry': User IDs for which a UserData object could not be created.
            'without_data': User IDs with no data.
            'no_identifiers': User IDs with no identifiers files.
            'irregular_directories': User IDs with unregistered survey data from irregular directories.
            'multiple_devices': User IDs with more than one associated device.
            'unknown_os': User IDs with data from both OS or unknown OS.
            'unnamed_objects': Object identifiers that don't have default names.
        summary (Summary):  Project overview for printing.
        info (OrderedDict):  Some organized information about the project.
    '''
    @classmethod
    def create(cls, raw_dirs, user_ids = 'all', 
               configuration = None, UTC_range = None,
               user_names = {}):
        '''
        Create a new BeiwePoject.

        Args:
            raw_dirs (str or list):  
                Paths to directories that may contain raw data from this user.
            user_ids (str or list):
                If 'all' then all available users are added to the project.
                Otherwise, one or more user ids to include.                
            configuration (Nonetype or str or list or dict): Optional.
                Can be None, or:
                - Path to a configuration file,
                - List of paths to configuration files,
                - Dictionary:
                    Keys are user_ids,
                    Values are lists of configuration files.
            UTC_range (Nonetype or list or dict): Optional.  Can be:
                - None, in which case all available data are registered.                
                - Ordered pair of date/times in filename_time_format, [start, end].
                  Files before start and after end are ignored.
                - Dictionary of user IDs (keys) and date/time pairs (values).
            user_names (dict): Optional. 
                Keys are user IDs, values are human-readable identifiers.
                If not empty, will be set as default user names.
            
        Returns:
            self (BeiweProject)
        '''
        self = cls.__new__(cls)        
        # format directories and ID lists
        if isinstance(raw_dirs, str): raw_dirs = [raw_dirs]
        self.raw_dirs = raw_dirs
        available_ids = list(set(join_lists([os.listdir(d) for d in self.raw_dirs])))
        available_ids.sort()
        if user_ids == 'all': user_ids = available_ids            
        elif isinstance(user_ids, str):
            user_ids = [user_ids]
        # set up lists
        self.lists = OrderedDict(zip(['iOS', 'Android'], [[], []]))
        # configuration and UTC range dictionaries
        if configuration is None: configuration = []
        if UTC_range is None: UTC_range = []
        self.lookup = OrderedDict(zip(['configuration', 'UTC_range'], 
                                      [coerce_to_dict(configuration, user_ids), 
                                       coerce_to_dict(UTC_range, user_ids)]))                
        self.load_configurations()
        # user name dictionary
        self.lookup['default_name'] = OrderedDict(user_names)
        # get object names from configuration files
        self.get_names(user_ids)
        # get user data registries
        data_range = []
        passive = []
        surveys = OrderedDict()
        self.data = OrderedDict()
        self.lookup['os'] = OrderedDict()                
        flag_labels = ['ignored_users', 'no_registry', 'without_data', 
                       'no_identifiers', 'irregular_directories', 
                       'multiple_devices', 'unknown_os', 'unnamed_objects']
        self.flags = OrderedDict(zip(flag_labels, 
                                     [[] for j in flag_labels]))
        for i in available_ids:
            if not i in user_ids:
                self.flags['ignored_users'].append(i)
            else:
                try:
                    temp = UserData.create(i, raw_dirs,
                                           self.lookup['UTC_range'][i],
                                           self.lookup['default_name'],
                                           self.lookup['object_name'])
                    self.data[i] = temp
                except:
                    logger.warning('Unable to create registry for %s.' % i)
                    self.flags['no_registry'].append(i)
        # update records
        for i in self.data:
            temp = self.data[i]
            if not temp.first is None: data_range += [temp.first, temp.last]
            # get device and OS info
            d = temp.device
            self.lookup['os'][i] = d.os
            if d.os in ['iOS', 'Android']: self.lists[d.os].append(i)
            else: self.flags['unknown_os'].append(i)
            if d.unique > 1: self.flags['multiple_devices'].append(i)
            # get registry info
            info = temp.info
            if info['raw_file_count'] == 0: 
                self.flags['without_data'].append(i)
            if info['irregular_directories'] > 0: 
                self.flags['irregular_directories'].append(i)
            # get passive registry
            p = temp.passive
            if p['identifiers']['count'] == 0:
                self.flag['no_identifiers'].append(i)
            passive += [k for k in p.keys() if p[k]['count'] > 0] 
            # get survey registry
            s = temp.surveys
            for k in s:
                sids = list(s[k]['ids'].keys())
                if len(sids) > 0:
                    if k in surveys: surveys[k] += sids                          
                    else: surveys[k] = sids
        # bookkeeping
        data_range.sort()
        if len(data_range) > 0:
            self.first, self.last = data_range[0], data_range[-1]
        else:
            self.first, self.last = None, None
        self.passive = sorted(list(set(passive)))
        self.surveys = OrderedDict()        
        for k in surveys:
            self.surveys[k] = sorted(list(set(surveys[k])))
        # sort user ids
        have_ids = list(self.data.keys())
        self.ids = sort_by(have_ids, [str(self.data[i].first) + str(self.data[i].last) for i in have_ids])
        # get default names and summarize
        if len(self.lookup['default_name']) == 0:
            temp = OrderedDict()
            n_ids = len(self.ids)  
            n_digits = len(str(n_ids))
            for j in range(n_ids):
                i = self.ids[j]
                count = str(j+1).zfill(n_digits)
                temp[i] = 'Participant ' + count            
            self.update_names(user_names = temp, object_names = None)
        else:
            self.summarize()
        logging.info('Finished generating study records for %d of %d users.' % (len(self.data), len(user_ids)))
        return(self)

    def update_configurations(self, configurations):
        '''
        Set new configuration files for project.
        Overwrites old configuration files.
        '''
        self.lookup['configuration'] = coerce_to_dict(configurations, self.ids)
        self.load_configurations()
        self.get_names()
        self.update_names()
    
    def load_configurations(self):
        '''
        Create a BeiweConfig object for each configuration file path.
        '''
        temp = OrderedDict()
        for k in self.lookup['configuration']:
            for p in self.lookup['configuration'][k]:
                if not p in temp:
                    try: temp[p] = BeiweConfig(p) 
                    except: logger.warning('Unable to read configuration file: %s' % os.path.basename(p))
        self.configurations = temp
        logger.info('Loaded configuration files.')
        
    def get_names(self, user_ids = None):
        '''
        Read configuration files. Get study names and object names.
        '''
        if user_ids is None: user_ids = self.ids
        # user name dictionary
        temp = OrderedDict()
        for i in self.lookup['configuration']:
            config_paths = self.lookup['configuration'][i]
            for p in config_paths:
                try: temp[i] = self.configurations[p].name
                except: temp[i] = None 
        self.lookup['study_name'] = temp
        if len(temp) > 0:
            logger.info('Finished reading study names.')
        # object name dictionary
        all_configs = join_lists([self.lookup['configuration'][i] for i in user_ids])
        all_configs = list(OrderedDict.fromkeys(all_configs)) # drop duplicates but keep order
        temp = OrderedDict()
        for c in all_configs: temp.update(self.configurations[c].name_assignments)
        if len(temp) != len(set(temp.values())):
            logger.warning('Object names are not unique; no object names are assigned.')
            self.lookup['object_name'] = OrderedDict()        
        else:
            self.lookup['object_name'] = temp
            if len(temp) > 0:
                logger.info('Finished reading object names.')
        self.lookup['reverse_object_name'] = {v:k for k, v in self.lookup['object_name'].items()}
        
    def update_names(self, user_names = None, object_names = None):
        if user_names   is not None: self.lookup['default_name'] = user_names
        if object_names is not None: self.lookup['object_name' ] = object_names            
        self.flags['unnamed_objects'] = []
        for i in self.data:
            self.data[i].summarize(self.lookup['default_name'], 
                                   self.lookup['object_name'])        
        self.summarize()            
        logger.info('Updated user name assignments.')

    def summarize(self, object_names = 'object_name'):
        '''
        Generate human-readable summaries.
        '''
        if isinstance(object_names, str): names = self.lookup[object_names]
        else: names = object_names
        reformat, observation_days, unit = summarize_UTC_range([self.first, self.last])        
        first, last = reformat
        unique_study_names = sorted(list(set(list(self.lookup['study_name'].values()))))
        if len(unique_study_names) > 0: studies_text = '\n'.join(unique_study_names)
        else: studies_text = None
        if len(unique_study_names) > 1: studies_text = studies_text
        overview = Summary(['Unique Beiwe Users', 'Study Name(s)', 'Raw Data Directories', 
                             'First Observation', 'Last  Observation', 'Project Duration'],
                           [len(self.ids), studies_text, len(self.raw_dirs), 
                            first, last, str(round(observation_days, 1)) + ' days'])
        devices = Summary(['iPhone Users', 'Android Users'], 
                          [len(self.lists['iOS']), len(self.lists['Android'])])
        flags = Summary(list(self.flags.keys()), 
                        [len(v) for v in list(self.flags.values())])
        user_info_keys = ['raw_file_count', 'size_bytes', 
                          'irregular_directories', 'unregistered_files']
        totals = []
        for k in user_info_keys:
            totals.append(sum([self.data[i].info[k] for i in self.ids]))
        totals[1] = naturalsize(totals[1])        
        registry = Summary(['Raw Files', 'Storage', 'Irregular Directories', 
                        'Unregistered Files'], totals)        
        passive, s_lab, s_txt = data_to_text(self.passive, self.surveys, self.data, 
                                                  self.lookup['object_name'])
        survey = Summary(s_lab, s_txt)
        self.summary = Summary(['Overview', 'Device Summary', 'Registry Summary', 
                                'Passive Data', 'Survey Data', 'Flagged Identifiers'], 
                               [overview, devices, registry, 
                                passive, survey, flags])
           
    @classmethod
    def load(cls, directory):
        '''
        Load an exported BeiweProject from json files.

        Args:
            directory (str): Path to directory with an exported BeiweProject.
            
        Returns:
            self (BeiweProject)
        '''
        self = cls.__new__(cls)        
        load_manage(self, directory)
        registry_dir = os.path.join(directory, 'records', 'registries')
        self.data = OrderedDict()
        load_ids = sorted(self.ids)
        for i in load_ids:
            filepath = os.path.join(registry_dir, i + '_registry.json')
            temp = UserData.load(filepath, 
                                 user_names =   self.lookup['default_name'], 
                                 object_names = self.lookup['object_name'])
            self.data[i] = temp
        self.load_configurations()
        self.summarize()
        return(self)

    def export(self, name, directory, track_time = True):
        '''
        Save json files with study records.  
        Overwrites pre-existing records.
        
        Args:
            name (str): Save object files to a folder with this name.
            directory (str): Where to save folder of records.
            track_time (bool): If True, export to a timestamped sub-directory.
            
        Returns:
            directory (str): Path to exported records.
        '''   
        directory = os.path.join(directory, name)
        if track_time:
            temp = 'project export from ' + local_now()
            directory = os.path.join(directory, temp.replace(' ', '_'))
        export_manage(self, directory)
        return(directory)
        
    def assemble(self, streams, user_ids = 'all'):
        '''
        Get a single dictionary with paths to all users' files for given streams.
        '''
        if user_ids == 'all': have_ids = self.ids
        else: have_ids = [i for i in user_ids if i in self.ids]        
        if isinstance(streams, str):
            if streams == 'passive': 
                streams = self.passive
            elif streams == 'surveys':
                temp = []
                for k in self.surveys: 
                    temp += [(k, sid) for sid in self.surveys[k]]
                streams = temp
            elif streams in self.surveys:
                k = streams
                streams = [(k, sid) for sid in self.surveys[k]]
            else: streams = [streams]
        a = OrderedDict.fromkeys(have_ids)
        for i in a:
            a[i] = self.data[i].assemble(streams)
        return(a)        

    def settings(self, setting, user_ids = 'all'):
        '''
        Get a dictionary with a configuration setting for each user.
        If more than one configuration file, use the first.
        '''
        if user_ids == 'all': have_ids = self.ids
        else: have_ids = [i for i in user_ids if i in self.ids]        
        s = OrderedDict.fromkeys(have_ids)
        for i in have_ids:
            try:
                bc = self.configurations[self.lookup['configuration'][i][0]]
                s[i] = bc.settings.passive[setting]
            except:
                logger.warning('User %s doesn\'t have setting \'%s\'.' % (i, setting))
        return(s)        
        
    def plot():
        pass
        
    def __eq__(self, other):
        return(check_same(self, other, to_check = 'all'))


class UserData():
    '''
    Class for organizing a user's raw Beiwe data.
        
    Attributes:
        id (str):  Beiwe user id.
        passive (OrderedDict):  Keys are passive data streams.
            Each value is an ordered dictionary with keys and values:
                'flag':  Values may be:
                    'not available for OS' - Data stream doesn't exist for this device type.
                    'not found' - Data stream directory is missing or empty.
                    None - Data stream exists for the device and is not empty.
                'count': Number of files for this data stream (int).
                'bytes': Total size of files on disk in bytes (int).                
                'files': List of all available files for the data stream.
        surveys (OrderedDict):  
            Keys are names of survey directories (e.g. 'audio_recordings', 'survey_timings').
            Each value is an ordered dictionary with keys and values:
                'flag':  None or 'not found'.
                'ids': 
                    An ordered dictionary. Keys are survey identifiers.  
                    Each value is an ordered dictionary with keys and values:
                        'count': Number of files for this data stream (int).
                        'bytes':    Total size of files on disk in bytes (int).                
                        'files': List of all available files for the corresponding survey.
        first, last (str):  
            Date/time of first and last observations.
            Formatted as '%Y-%m-%d %H_%M_%S'.
        UTC_range (list or Nonetype):  
            Ordered pair of date/times in filename_time_format, [start, end].
            If not None, files before start and after end were ignored.            
        not_registered (list): 
            Paths to unregistered files in irregular directories.
            Irregular directories are survey directories that contain raw data files.
        device (DeviceInfo): Represents contents of the user's identifier files.
        summary (Summary): Overview of user data for printing.
        info (OrderedDict): See headers.info_header for details.
    '''
    @classmethod
    def create(cls, user_id, raw_dirs, UTC_range = None,
               user_names = {}, object_names = {}):
        '''
        Generate user registry from directories of raw Beiwe data.

        Args:
            user_id (str) = Beiwe user ID.
            raw_dirs (str or list):  
                Paths to directories that may contain raw data from this user.
            UTC_range (list or Nonetype): Optional.  
                Ordered pair of date/times in filename_time_format, [start, end].
                If not None, ignore files before start and after end.
            user_names, object_names (dict):
                Optional dictionaries with name assignments.

        Returns:
            self (UserData)
        '''
        self = cls.__new__(cls) 
        self.id = user_id
        if UTC_range == []: UTC_range = None
        self.UTC_range = UTC_range
        if isinstance(raw_dirs, str): raw_dirs = [raw_dirs]
        data_range = []
        # get identifiers and device
        self.passive = OrderedDict()
        self.passive['identifiers'] = identifiers_registry(self.id, raw_dirs, self.UTC_range)        
        self.device = DeviceInfo(self.passive['identifiers']['files'])
        phone_os = self.device.os
        if phone_os is None:
            logger.warning('Unable to get device info for ' + self.id + '.')
            phone_os = 'both'
        # get passive data registry
        data_range, pr = passive_registry(self.id, phone_os, raw_dirs, self.UTC_range)
        self.passive.update(pr)
        # get survey data registry
        survey_range, self.surveys, self.not_registered = survey_registry(self.id, raw_dirs, self.UTC_range)
        data_range += survey_range
        # get first & last observation datetimes
        data_range.sort()        
        if len(data_range) > 0:
            self.first = data_range[0]
            self.last = data_range[-1]
        else: self.first, self.last = None, None
        # get summary
        self.summarize(user_names, object_names)
        logger.info('Created raw data registry for Beiwe user ID %s.' % self.id)
        return(self)

    def summarize(self, user_names, object_names, ndigits = 1):
        '''
        Collect some information and summary stats.
        
        Args:
            user_names, object_names (dict):
                Dictionaries with name assignments.
            ndigits (int): Number of digits for rounding.
            
        Returns:
            None            
        '''
        try: user_name = user_names[self.id]
        except: user_name = None
        # followup
        reformat, project_days, unit = summarize_UTC_range(self.UTC_range)        
        begin, end = reformat        
        # observations
        reformat, observation_days, unit = summarize_UTC_range([self.first, self.last])        
        first, last = reformat
        # summarize raw files
        raw_count, size = 0, 0
        for k in self.passive:
            raw_count += self.passive[k]['count']
            size += self.passive[k]['bytes']
        # irregular directories
        irrds = list(set([os.path.dirname(p) for p in self.not_registered]))
        irregular_directories = len(irrds)
        # get overview        
        info = [self.id, user_name, 
                begin, end, project_days, 
                first, last, observation_days,
                raw_count, size, self.device.unique, self.device.os,
                irregular_directories, len(self.not_registered)]
        self.info = OrderedDict(zip(info_header, info))
        # get summary
        labels = ['Identifiers']
        items = [Summary(['Beiwe User ID', 'User Name'], [self.id, user_name])]
        if not self.UTC_range is None:
            project = Summary(['Begin', 'End', 'Duration'],
                              [begin, end, str(project_days) + ' days'])
            labels.append('Project Range')
            items.append(project)
        observations = Summary(['First Observation', 'Last  Observation', 
                                'Observation Period', 'Raw Files', 'Storage'],
                               [first, last, str(observation_days) + ' days', 
                                raw_count, naturalsize(size)])
        device = Summary(['Number of Phones', 'Phone OS'],
                         [self.device.unique, self.device.os])
        registry = Summary(['Irregular Directories', 'Unregistered Files'],
                           [irregular_directories, len(self.not_registered)])
        labels += ['Raw Data Summary', 'Device Records', 'Registry Issues']
        items += [observations, device, registry]
        pt, s_types, s_text = registry_to_text(self.passive, self.surveys, 
                                               self.first, self.last, object_names)
        labels += ['Passive Data Summary', 'Survey Data Summary']
        items += [pt, Summary(s_types, s_text)]
        self.summary = Summary(labels, items)

    @classmethod
    def load(cls, path, user_names = {}, object_names = {}):
        '''
        Load user registry from a json file.
        
        Args:
            path (str):  
                Path to an exported UserData object.
            user_names, object_names (dict):
                Optional dictionaries with name assignments.
            
        Returns:
            self (UserData)
        '''
        self = cls.__new__(cls)
        load_manage(self, path)
        self.device = DeviceInfo(self.passive['identifiers']['files'])
        self.summarize(user_names, object_names)
        logger.info('Loaded raw data registry for Beiwe user ID %s.' % self.id)
        return(self)

    def export(self, directory):
        '''
        Saves record of merged file paths. 
        
        Args:
            directory (str):  Directory where json file should be saved.
        '''
        export_manage(self, directory)

    def assemble(self, streams):
        '''
        Get a dictionary with paths to user's files for given list of data streams.
        An item in streams can be: 
            a passive data stream (str), 
            an ordered pair(survey type, survey identifier) (tuple).
        '''
        if isinstance(streams, str): streams = [streams]
        if isinstance(streams, tuple): streams = [streams]
        a = OrderedDict()
        for s in streams:
            if isinstance(s, str):
                if s in self.passive.keys(): a[s] = self.passive[s]['files']
                else: a[s] = []
            elif isinstance(s, tuple):
                s_type, sid = s
                if sid in self.surveys[s_type]['ids']:
                    a[s] = self.surveys[s_type]['ids'][sid]['files']
                else: a[s] = []
            else: logger.warning('Check stream format; %s is neither a string nor a tuple.' % str(s))
        return(a)   

    def __eq__(self, other):
        return(check_same(self, other, to_check = 'all'))


class DeviceInfo():
    '''
    Class for reading identifier files from raw Beiwe data.

    Args:  
        paths (str or list):  One or more paths to the user's identifier files.
    
    Attributes:
        identifiers (OrderedDict):  
            Keys are paths to identifier files.  
            Values are OrderedDicts that contain information from the corresponding identifier file.
        os (str): 
            'iOS' or 'Android' if history includes only one OS.
            Otherwise 'both'.
        unique (int): Number of unique devices that were used during followup.
    '''
    def __init__(self, paths = []):
        if type(paths) is str:
            paths = [paths]
        # sort paths according to file creation date
        paths = sort_by(paths, [os.path.basename(p) for p in paths])
        # read identifiers in order of creation
        self.identifiers = OrderedDict()
        for p in paths:
            f = open(p, 'r')
            lines = list(f)
            f.close()
            keys = lines[0].replace('\n', '').split(',')
            values = lines[1].split(',')
            # iPhone identifier files have an extra comma.
            # Replace comma with an underscore:
            if len(values) > len(keys):
                values = values[:-2] + ['_'.join(values[-2:])]        
            # Check header
            if keys != identifiers_header:
                logger.warning('Unknown identifiers header for user ID %s' % values[2])
            self.identifiers[p] = OrderedDict(zip(keys, values))  
        # get os
        os_history = self.history('device_os')
        if len(os_history) == 0: 
            logger.warning('Initialized empty DeviceInfo object.')
            self.os = None
        elif len(list(set(os_history.values()))) == 1:
            self.os = list(os_history.values())[0]
        else:
            self.os = 'both'
            logger.warning('Found multiple operating systems for user ID %s.' % values[2])
        # get device count
        device_history = self.history('device_id')
        if len(device_history) == 0:
            self.unique = None
        else:
            self.unique =  len(list(set(device_history.values())))
            if self.unique > 1:
                logger.warning('Found multiple devices for user ID %s.' % values[2])

    def history(self, header):
        '''
        Return a dictionary with history of a particular device attribute.
        For older files, replaces 'iPhone OS' with 'iOS' under header 'device_os'.
        
        Args:
            header (str): Column header from identifiers CSV.
                e.g. 'device_os', 'beiwe_version'

        Returns:
            history (OrderedDict):
                Keys are sorted timestamps.
                Values are device attributes observed at those times.        
        '''
        history = OrderedDict()
        if not header in identifiers_header:
            logger.warning('%s isn\'t a device attribute.' % header)
        else:
            for d in self.identifiers.values():
                history[d['timestamp']] = d[header]
        # operating system history
        if header == 'device_os':
            for k in history.keys():
                if history[k] == 'iPhone OS': history[k] = 'iOS'
        return(history)
        
    def export(self, directory):
        '''
        Write all identifier information to a single csv.
        '''
        export_manage(self, directory)

            
    def __eq__(self, other):
        return(check_same(self, other, to_check = 'all'))
