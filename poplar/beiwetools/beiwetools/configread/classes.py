'''Classes for working with Beiwe study configurations.

See /examples/configread_example.ipynb for sample usage.
'''
import os
import logging
from collections import OrderedDict

from beiwetools.helpers.time import local_now
from beiwetools.helpers.classes import Summary
from beiwetools.helpers.functions import (check_same, setup_directories, 
                                          read_json, write_json, write_string)

from .surveys import survey_classes, BeiweSurvey
from .functions import load_settings, study_settings


logger = logging.getLogger(__name__)


class BeiweConfig():
    """
    Class for representing a Beiwe study configuration as encoded in a JSON file.

    Args:
        path (str): Path to the JSON configuration file.
            Or a path to a directory containing a previously exported configuration.
        names_path (str):  Path to a JSON file with name assignments for study objects.
            Optional. If None: 
            - study_name will be taken from the name of the configuration file,
            - survey and question names will be assigned in the order they appear in the JSON file.

    Attributes:
        warnings (list): Records of unknown settings or object types, if any.
        name (str): 
            An additional identifier for the study, such as the name of the configuration file.
        paths (str): Paths to the files used to create the BeiweConfig object.
        name_assignments (OrderedDict): 
            Keys are object identifiers, values are readable names.
            Default assignments are:
                "Survey_1", "Survey_2", etc.
                "Survey_1_Question_1", "Survey_1_Question_2", etc.
        identifier_lookup (dict): 
            Keys are names and values are object identifiers.
            This dictionary is just for convenience, to find an identifier given the name of a survey or question.
        default_names (bool): 
            Whether or not surveys & questions are assigned default names.
            True if configuration is loaded with names_path = None.
            False if:
                The update_names() method has been called,
                The configuration is loaded from an export,
                Or configuration is loaded with a name assignment dictionary.
        raw (OrderedDict): The deserialized configuration file.
        extended_format (bool): 
            True if the JSON file uses MongoDB Extended JSON format, False otherwise.
        identifier (int or str): 
            If the JSON file uses MongoDB Extended JSON format, then the id is a 24-character identifier for the study.
            Otherwise, the id is an integer.
            Note that this id may not be the same as the backend's identifier for this study.
        summary (Summary):  Some features of the BeiweConfig object for printing.
        device_settings (OrderedDict): Keys are setting names, values are the settings.    
        surveys (OrderedDict): Keys are survey IDs, values are BeiweSurvey objects.
        survey_ids (OrderedDict): 
            Keys are survey class adjectives, e.g. 'audio', 'tracking', 'other'.
            Values are lists of identifiers for surveys of that class.
        to_check (list): Attributes to consider when checking equality.
    """    
    def __init__(self, path, names_path = None):
        self.warnings = []
        self.paths = OrderedDict()
        # check if path is a directory:
        if os.path.isdir(path):
            records = os.path.join(path, 'records')
            path = os.path.join(records, 'raw.json')
            names_path = os.path.join(records, 'names.json')
        self.paths['path'] = path
        self.paths['names_path'] = names_path
        self.default_names = self.paths['names_path'] is None
        # load configuration file
        self.raw = read_json(path)
        # get names
        self.name_assignments = OrderedDict()
        if not self.default_names:
            self.name_assignments = read_json(names_path)
        # does the configuration file use MongoDB Extended JSON? 
        self.extended_format = '$oid' in str(self.raw)
        # get identifier and deleted status
        if self.extended_format:
            self.identifier = self.raw['device_settings']['_id']['$oid']
        else:
            self.identifier = self.raw['device_settings']['id']
        if isinstance(self.identifier, int): self.identifier = str(self.identifier)
        # get study name
        if self.identifier in self.name_assignments:
            self.name = self.name_assignments[self.identifier]
        else:
            self.name = os.path.basename(path).replace('_surveys_and_settings', '').replace('.json', '').replace('_', ' ')
            self.name_assignments[self.identifier] = self.name
        # read settings
        self.settings = DeviceSettings(self.raw['device_settings'], self)
        # read surveys
        self.surveys = OrderedDict()
        self.survey_ids = OrderedDict()
        n_surveys = len(self.raw['surveys'])
        n_digits = len(str(n_surveys)) 
        for i in range(n_surveys):
            s = self.raw['surveys'][i]
            # get/assign name                
            if self.extended_format:
                survey_id = s['_id']['$oid']
            else:
                survey_id = s['object_id']                       
            if survey_id in self.name_assignments.keys():
                name = self.name_assignments[survey_id]
            else:
                name = 'Survey ' + str(i+1).zfill(n_digits)            
                self.name_assignments[survey_id] = name
            # read survey
            if s['survey_type'] in survey_classes.keys(): 
                SC = survey_classes[s['survey_type']]
                survey = SC(s, self, name)
            else:
                survey = BeiweSurvey(s, self, name)
                flag = 'Found unknown survey type: %s' % survey.type
                self.warnings.append(flag)
                logger.warning(flag)
            self.surveys[survey.identifier] = survey
            # update dictionary of survey ids
            if survey.class_adjective in self.survey_ids.keys():
                self.survey_ids[survey.class_adjective].append(survey.identifier)
            else:
                self.survey_ids[survey.class_adjective] = [survey.identifier]
        # get summary
        self.summarize()
        # identifier lookup dictionary            
        self.get_lookup()
        # when checking equality
        self.to_check = ['settings', 'surveys']

    def get_lookup(self):
        '''
        Invert the name_assignments dictionary.
        '''
        self.identifier_lookup = {v:k for k, v in self.name_assignments.items()}

    def summarize(self):
        '''
        Get a configuration overview for printing.
        '''
        identifiers = Summary(['Study Identifier (may not agree with backend)', 'MongDB Extended JSON format', 'Default names'],
                              [self.identifier, self.extended_format, self.default_names])
        active = OrderedDict()
        deleted = OrderedDict()
        for k in self.survey_ids.keys():
            a = [i for i in self.survey_ids[k] if not self.surveys[i].deleted is True]
            d = [i for i in self.survey_ids[k] if self.surveys[i].deleted is True]            
            active[k.capitalize()]  = len(a)
            deleted[k.capitalize()] = len(d)
        active_survey_counts  = Summary(list(active.keys()), list(active.values()))
        deleted_survey_counts = Summary(list(deleted.keys()), list(deleted.values()))
        self.summary = Summary(['Identifiers', 'Number of Active Surveys', 'Number of Deleted Surveys'],
                               [identifiers, active_survey_counts, deleted_survey_counts],
                               header = self.name)
    
    def update_names(self, new_names):
        '''
        Update all name assignments.

        Args:
            new_names (dict or OrderedDict): 
                Keys are old names, values are new names.
            
        Returns:
            None
        '''
        if len(list(set(self.name_assignments.values()))) != len(self.name_assignments):
            logger.warning('Name update failed.  Name assignments are not unique.')
        else:
            for k in self.name_assignments:
                if self.name_assignments[k] in new_names.keys():
                    self.name_assignments[k] = new_names[self.name_assignments[k]]
            self.default_names = False
            self.name = self.name_assignments[self.identifier]
            for s in self.surveys.values(): s.update_names(self.name_assignments)
            self.summarize()
            self.get_lookup()

    def __eq__(self, other):
        return(check_same(self, other, self.to_check))
                           
    def export(self, directory, track_time = True, indent = 4, max_width = 70):
        '''
        Write study documentation to text files.

        Args:
            directory (str):  Path to a directory.
            track_time (bool):  If True, nests output in a folder with current local time.
            indent (int):  Indentation for pretty printing.
            max_width (int):  Maximum line width for text file.
        
        Returns:
            out_dir (str): Path to study documentation directory.
        '''       
        if track_time:
            temp_name = 'configuration documentation from ' + local_now()
            temp = os.path.join(directory, temp_name.replace(' ', '_'))
        else:
            temp = directory
        out_dir = os.path.join(temp, self.name.replace(' ', '_'))
        # export settings
        settings_dir = os.path.join(out_dir, 'settings')
        config_dir = os.path.join(out_dir, 'records')
        setup_directories([out_dir, settings_dir, config_dir])
        # export configuration records
        write_json(self.raw, 'raw', config_dir)
        write_json(self.paths, 'paths', config_dir)        
        write_json(self.name_assignments, 'names', config_dir)
        # export summary
        self.summary.to_file('overview', out_dir, indent = indent, max_width = max_width)
        # export warnings        
        write_string('', 'warnings', out_dir, title = 'Warnings:', mode = 'w')
        if len(self.warnings) == 0: write_string('None', 'warnings', out_dir, mode = 'a')
        else:
            for w in self.warnings: write_string(w, 'warnings', out_dir, mode = 'a')
        # export device settings
        self.settings.general_summary.to_file('general_settings', settings_dir, 
                                              indent = indent, max_width = max_width)
        self.settings.passive_summary.to_file('passive_data_settings', settings_dir, 
                                              indent = indent, max_width = max_width)        
        self.settings.display_summary.to_file('display_settings', settings_dir, 
                                              indent = indent, max_width = max_width,
                                              extra_breaks = [1, 2])        
        # export surveys        
        surveys = list(self.surveys.values())
        adjectives = list(set([s.class_adjective for s in surveys]))
        survey_dirs = [os.path.join(out_dir, a + '_surveys') for a in adjectives]
        setup_directories(survey_dirs)
        for s in surveys:
            survey_dir = os.path.join(out_dir, s.class_adjective + '_surveys')
            name = s.name.replace(' ', '_')
            if s.deleted is True: name = 'deleted_' + name
            s.summary.to_file(name, survey_dir, indent = indent, max_width = max_width)
        # return location of documentation
        return(out_dir)


class DeviceSettings():
    '''
    Class for representing device settings from a Beiwe configuration file.
    
    Args:
        device_settings (OrderedDict):  From a JSON configuration file.
        beiweconfig (BeiweConfig): The instance to which these settings belong.
        
    Attributes:
        raw (OrderedDict): Deserialized settings.
        identifiers (OrderedDict): Study identifiers.
        deleted (OrderedDict): Whether or not the study was deleted.
        survey (OrderedDict): Global survey settings.
        app (OrderedDict): Settings for app behavior.
        display (OrderedDict): Text displayed by the app.
        passive (OrderedDict): Passive data settings.
        other (OrderedDict): Settings that haven't been documented.   
        general_summary, passive_summary, display_summary (Summary):
            Organized settings for printing or export.
        to_check (list): Attributes to consider when checking equality.
    '''
    def __init__(self, device_settings, beiweconfig):
        self.raw = device_settings
        # get settings        
        self.identifiers =  load_settings(study_settings['Identifiers'], self.raw)
        if beiweconfig.extended_format: 
            self.identifiers['_id'] = self.identifiers['_id']['$oid'] 
        self.deleted =      load_settings(study_settings['Deleted Status'], self.raw)
        self.survey =       load_settings(study_settings['Survey Settings'], self.raw)
        self.app =          load_settings(study_settings['App Settings'], self.raw)
        self.display =      load_settings(study_settings['Display Settings'], self.raw)
        self.passive =      load_settings(study_settings['Passive Data Settings'], self.raw)
        known_settings = [k for s in list(study_settings.values()) for k in s]
        unknown_settings = [k for k in self.raw.keys() if k not in known_settings]
        if len(unknown_settings) > 0:
            self.other = load_settings(unknown_settings, self.raw)
            for u in unknown_settings:
                flag = 'Found unknown device setting: %s' % u
                beiweconfig.warnings.append(flag)
                logger.warning(flag)
        else:
            self.other = None
        # summaries
        self.general_summary = Summary(list(study_settings.keys())[:4] + ['Other/Unknown Settings'],
                                       [self.identifiers, self.deleted, self.survey, self.app, self.other])
        self.passive_summary = Summary(['Passive Data Settings'], [self.passive])
        self.display_summary = Summary(['Display Settings'], [self.display], sep = ':\n ')
        # when checking equality
        self.to_check = ['survey', 'app', 'display', 'passive', 'other']

    def __eq__(self, other):
        return(check_same(self, other, self.to_check))