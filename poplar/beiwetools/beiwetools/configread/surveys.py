''' Classes for representing surveys found in Beiwe studies.

'''
import logging
from collections import OrderedDict
from beiwetools.helpers.classes import Summary
from beiwetools.helpers.functions import check_same
from .functions import load_settings, load_timings, survey_settings
from .questions import tracking_questions, TrackingQuestion


logger = logging.getLogger(__name__)


class BeiweSurvey():
    '''
    Class for representing a survey from a Beiwe configuration file.
    
    Args:
        survey_config (OrderedDict):  From a JSON configuration file.
        beiweconfig (BeiweConfig): The instance to which this survey belongs.
        
    Attributes:
        name (str): Human-readable identifier for the study.
        class_adjective (str): 
            Short description of the survey type, e.g. 'audio', 'tracking', 'other'.
            This is used for documenting and organizing surveys.        
        raw (OrderedDict): The deserialized survey configuration.
        extended_format (bool): 
            True if the JSON file uses MongoDB Extended JSON format, False otherwise.
        deleted (bool):  True if the survey was deleted.
        type (str):  Examples are 'audio_survey', 'dummy', or 'tracking_survey'.        
        identifier (str):  The 24-character identifier for the survey.
        settings (OrderedDict):  Survey settings from the JSON configuration file.
        timings (OrderedDict): Human-readable timings.  
            Keys are days of the week, values are lists of scheduled times. 
        content (list):  Content from the JSON configuration file.       
        content_dict (OrderedDict): Content organized for printing export.
        summary (Summary): Summary of survey attributes for printing.
        to_check (list): Attributes to consider when checking equality.
    '''
    def __init__(self, survey_config, beiweconfig, name):
        self.name = name
        self.class_adjective = 'other'
        self.raw = survey_config
        # read common survey settings
        info = load_settings(survey_settings['common_survey_info'], self.raw)
        # get identifier and deleted status
        if beiweconfig.extended_format:
            self.identifier = info['_id']['$oid']
            self.deleted = None
        else:
            self.identifier = info['object_id']
            self.deleted = info['deleted']
        # get survey type
        self.type = info['survey_type']
        # check for unknown attributes
        known = survey_settings['common_survey_info'] + ['settings', 'timings', 'content']
        for k in self.raw.keys():
            if k not in known:
                flag = 'Found unknown survey attribute for %s: %s' % (self.type, k)
                beiweconfig.warnings.append(flag)
                logger.warning(flag)
        # get settings
        self.get_settings()
        # get timings
        self.timings = load_timings(self.raw['timings'])
        # get content
        self.get_content(beiweconfig)
        # summary
        self.summarize()

    def get_settings(self):
        self.settings = self.raw['settings']
        
    def get_content(self, beiweconfig):
        self.content = self.raw['content']
        self.content_dict = OrderedDict([('Content', self.content)])
        # when checking equality
        self.to_check = ['type', 'settings', 'timings', 'content']

    def summarize(self):
        info = Summary(['Identifier', 'Type', 'Deleted'],
                       [self.identifier, self.type, self.deleted])
        labels = ['Info', 'Settings', 'Timings'] + list(self.content_dict.keys())
        items = [info, self.settings, self.timings] + list(self.content_dict.values())
        self.summary = Summary(labels, items, header = self.name)

    def update_names(self, name_assignments):
        self.name = name_assignments[self.identifier]
        self.summarize()

    def __eq__(self, other):
        return(check_same(self, other, self.to_check))

        
class AudioSurvey(BeiweSurvey):
    '''
    Class for representing an audio survey from a Beiwe configuration file.
    Inherits from BeiweSurvey.
        
    Attributes:
        settings (OrderedDict):  
            Audio survey settings from the JSON configuration file.
        prompt (str):  Audio survey prompt.
    '''    
    def __init__(self, survey_config, beiweconfig, name):
        super().__init__(survey_config, beiweconfig, name)
        self.class_adjective = 'audio'
        
    def get_settings(self):
        self.settings = load_settings(survey_settings['audio_survey'], 
                                      self.raw['settings'])
        
    def get_content(self, beiweconfig):
        try:
            self.prompt = self.raw['content'][0]['prompt']
        except:
            self.prompt = None
        warning = False
        if len(self.raw['content']) > 1: warning = True
        if len(self.raw['content']) == 1:
            if len(self.raw['content'][0].keys()) > 1:
                warning = True
        if warning:
            flag = 'Found unknown content for audio survey id %s' % self.identifier
            beiweconfig.warnings.append(flag)
            logger.warning(flag)
        self.content_dict = OrderedDict([('Prompt', self.prompt)])        
        # when checking equality
        self.to_check = ['type', 'settings', 'timings', 'prompt']     


class TrackingSurvey(BeiweSurvey):        
    '''
    Class for representing a tracking survey from a Beiwe configuration file.
    Inherits from BeiweSurvey.
        
    Attributes:
        settings (OrderedDict):  
            Tracking survey settings from the JSON configuration file.
        questions (OrderedDict):  
            Keys are question IDs, values are Tracking Question objects.
    '''    
    def __init__(self, survey_config, beiweconfig, name):
        super().__init__(survey_config, beiweconfig, name)
        self.class_adjective = 'tracking'

    def get_settings(self):
        self.settings = load_settings(survey_settings['tracking_survey'], 
                                      self.raw['settings'])

    def get_content(self, beiweconfig):
        self.questions = OrderedDict()
        n_questions = len(self.raw['content'])
        n_digits = len(str(n_questions))
        for i in range(n_questions):
            question_config = self.raw['content'][i]
            qid = question_config['question_id']
            # get/assign name
            if qid in beiweconfig.name_assignments:
                qname = beiweconfig.name_assignments[qid]
            else:
                qname = self.name + ' - Question ' + str(i+1).zfill(n_digits)
                beiweconfig.name_assignments[qid] = qname
            # identify question type            
            qtype = question_config['question_type']
            if qtype in tracking_questions.keys():
                self.questions[qid] = tracking_questions[qtype](question_config, beiweconfig, qname)         
            else: 
                self.questions[qid] = TrackingQuestion(question_config, beiweconfig, qname)
                flag = 'Found unknown question type: %s' %qtype
                beiweconfig.warnings.append(flag)
                logger.warning(flag)        
        self.get_content_dict()
        # when checking equality
        self.to_check = ['type', 'settings', 'timings', 'questions']     

    def get_content_dict(self):
        question_summary = Summary(['\n' + q.name for q in self.questions.values()], 
                                       [q.summary for q in self.questions.values()])        
        self.content_dict = OrderedDict([('Questions', question_summary)]) 

    def update_names(self, name_assignments):
        self.name = name_assignments[self.identifier]
        for q in self.questions.values():
            q.name = name_assignments[q.identifier]
            q.summarize()
        self.get_content_dict()
        self.summarize()


# Dictionary of tracking survey types
stypes = ['audio_survey', 'tracking_survey']
sclasses = [AudioSurvey, TrackingSurvey]
survey_classes = OrderedDict(zip(stypes, sclasses))