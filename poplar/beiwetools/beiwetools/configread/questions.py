'''Classes for representing Beiwe tracking survey questions.

'''
import logging
from collections import OrderedDict
from beiwetools.helpers import check_same, Summary
from .functions import load_settings, question_settings
    

logger = logging.getLogger(__name__)


class TrackingQuestion():
    '''
    Class for representing a tracking survey question from a Beiwe configuration file.
    
    Args:
        question_config (OrderedDict):  From a JSON configuration file.
        beiweconfig (BeiweConfig): The instance to which this question belongs.
        
    Attributes:
        name (str): Human readable name for the question.
        class_name (str): Description of question type.
        raw (OrderedDict): The deserialized question configuration.        
        identifier (str):  The 36-character question_id. 
        type (str):  Question type, e.g 'info_text_box'
        text (str): Formatted text that is displayed with the question.
        other (OrderedDict):  Content that is specific to the question type.
        logic (OrderedDict):  The branching logic configuration from the JSON file.
        summary (Summary): Summary of question attributes for printing.
        to_check (list): Attributes to consider when checking equality.
    '''    
    def __init__(self, question_config, beiweconfig, name):
        self.raw = question_config
        self.name = name
        self.class_name = 'Generic Question'
        # get info
        info = load_settings(question_settings['common_question_info'], self.raw)
        # get identifier
        self.identifier = info['question_id']
        # get question type and display text
        self.type = info['question_type']
        self.text = info['question_text']
        # get logic
        if info['display_if'] == 'Not found':
            self.logic = None
        else:
            self.logic = info['display_if']
        # get other content
        self.get_other_content()        
        # check for unknown attributes
        if self.type in question_settings.keys():
            self.check_unknown_attributes(beiweconfig)
        # summary
        self.summarize()    
        # when checking equality
        self.to_check = ['type', 'logic', 'text', 'other']
        
    def get_other_content(self):
        self.other = OrderedDict()        
        for k in self.raw.keys():
            if k not in question_settings['common_question_info']:
                self.other[k] = self.raw[k]

    def check_unknown_attributes(self, beiweconfig):
        known = question_settings['common_question_info'] + question_settings[self.type]
        for k in self.raw.keys():
            if k not in known:
                flag = 'Found unknown question attribute for %s: %s' % (self.type, k)
                beiweconfig.warnings.append(flag)
                logger.warning(flag)
            
    def update_names(self, name_assignments):
        self.name = name_assignments[self.identifier]
        self.summarize()
        
    def summarize(self):
        if self.logic is None:
            logic_message = 'Does not use branching logic.' 
        else:
            logic_message = 'Uses branching logic, see configuration file for details.'
        info_labels = ['Identifier', 'Type', 'Logic', 'Text']
        info_content = [self.identifier, self.type, logic_message, self.text]
        labels = info_labels + list(self.other.keys())
        items =  info_content + list(self.other.values())
        self.summary = Summary(labels, items, header = self.name)

    def __eq__(self, other):
        return(check_same(self, other, self.to_check))

    
class InfoTextBox(TrackingQuestion):
    '''
    Class for representing an info text box from a Beiwe tracking survey.
    Inherits from TrackingQuestion.
    '''
    def __init__(self, question_config, beiweconfig, name):
        super().__init__(question_config, beiweconfig, name)
        self.class_name = 'Info Text'


class CheckBox(TrackingQuestion):
    '''
    Class for representing a check box question from a Beiwe tracking survey.
    Inherits from TrackingQuestion.

    Attributes:
        answers (list): 
            List of question answers in the order they appear in the configuration file.
        scores (list): 
            Integer scores for each answer.  
            By default, the index of the answer in the order that it appears in the configuration file.
    '''
    def __init__(self, question_config, beiweconfig, name):
        super().__init__(question_config, beiweconfig, name)
        self.class_name = 'Check Box'
        
    def get_other_content(self):
        self.answers = [list(a.values())[0] for a in self.raw['answers']]
        self.scores = list(range(len(self.answers)))
        numbered_answers = Summary([str(i) for i in self.scores], 
                                   self.answers)
        self.other = OrderedDict([('Answers', numbered_answers)])


class RadioButton(TrackingQuestion):
    '''
    Class for representing a radio button question from a Beiwe tracking survey.
    Inherits from TrackingQuestion.

    Attributes:
        answers (list): 
            List of question answers in the order they appear in the configuration file.
        scores (list): 
            Integer scores for each answer.  
            By default, the index of the answer in the order that it appears in the configuration file.
    '''
    def __init__(self, question_config, beiweconfig, name):
        super().__init__(question_config, beiweconfig, name)
        self.class_name = 'Radio Button'

    def get_other_content(self):
        self.answers = [list(a.values())[0] for a in self.raw['answers']]
        self.scores = list(range(len(self.answers)))
        numbered_answers = Summary([str(i) for i in self.scores], 
                                   self.answers)
        self.other = OrderedDict([('Answers', numbered_answers)])


class Slider(TrackingQuestion):
    '''
    Class for representing a slider question from a Beiwe tracking survey.
    Inherits from TrackingQuestion.
    
    Attributes:
        min, max (int): Slider endpoints.
    '''
    def __init__(self, question_config, beiweconfig, name):
        super().__init__(question_config, beiweconfig, name)
        self.class_name = 'Slider'
        
    def get_other_content(self):
        self.min = self.raw['min']
        self.max = self.raw['max']
        self.other = OrderedDict([('Min', self.min), ('Max', self.max)])       
        

class FreeResponse(TrackingQuestion):
    '''
    Class for representing a free response question from a Beiwe tracking survey.
    Inherits from TrackingQuestion.
    
    Attributes:
        text_field_type (str): Description of text field.
    '''
    def __init__(self, question_config, beiweconfig, name):
        super().__init__(question_config, beiweconfig, name)
        self.class_name = 'Free Response'

    def get_other_content(self):
        self.text_field_type = self.raw['text_field_type']
        self.other = OrderedDict([('Text Field Type', self.text_field_type)])         


# Dictionary of tracking question types
qtypes = ['info_text_box', 'checkbox', 'radio_button', 'slider', 'free_response']
qclasses = [InfoTextBox, CheckBox, RadioButton, Slider, FreeResponse]
tracking_questions = OrderedDict(zip(qtypes, qclasses))