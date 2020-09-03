'''Template for a generic class that can be exported and loaded.

    - Export/load methods use JSON format for human-readable representations.
    - Attributes should be serializable Python types.
    - Equivalence of instances (==) is based on equality of attributes. 
'''

import os
import json
from logging import getLogger


logger = getLogger(__name__)


class SerializableClass():
    '''
    Attributes:
        do_not_serialize (list): List of attributes (str) that should not be 
            serialized.
    '''
    def __init__(self):
        '''
        This method can be left empty.
        '''
        pass 

    @classmethod
    def create(cls):
        '''
        Use the create() method to initialize an instance with parameters, e.g.
            SerializableClass.create(**kwargs)
        '''
        self = cls.__new__(cls)
        # initialize self
        self.do_not_serialize = []
        return(self)
        
    def export(self, name, out_dir):
        '''
        Serialize object attributes and write to a JSON file.
        
        Args:
            name (str): Name for the JSON file, without extension.
            out_dir (str): Full path to directory where JSON file will be 
                written.

        Returns:
            class_file (str): Path to JSON with serialized attributes.
        '''
        try:
            all_attributes = self.__dict__.keys()
            select_attributes = [k for k in all_attributes if not k in self.do_not_serialize]
            to_write = {k:self.__dict__[k] for k in select_attributes}
            filepath = os.path.join(out_dir, name + '.json')
            json.dump(to_write, filepath, indent = 4)
        except:
        		logger.warning('Unable to write JSON file.')
    
    @classmethod
    def load(cls, attributes_filepath):
        '''
        Given a file with serialized object attributes, initialize a new class
        instance with those attributes.

        Args:
            class_file (str): Full path to a file with serialized object
                attributes.
            
        Returns:
            class_instance (SerializableClass): An instance of the class with
                the deserialized attributes.            
        '''        
        try:
            # initialize object
            self = cls.__new__(cls)
            self.__dict__ = json.load(attributes_filepath)
            return(self)
        except:
        		logger.warning('Unable to load JSON file.')
                    
    def __eq__(self, other):
        '''
        Define equivalence (==).
        '''
        same = False
        if type(other) == type(self):        
            try:
                same = other.__dict__ == self.__dict__
            except:
                logger.warning('Equivalence is not implemented.')                
        return(same)