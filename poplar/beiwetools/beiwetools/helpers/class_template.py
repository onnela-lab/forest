'''Template for a generic class that can be exported and loaded.

    - Export/load class using JSON format for human-readable representations.
    - Equivalence of instances (==) is based on equality of attributes. 
    - Attributes should be JSON-serializable Python objects, e.g. lists, 
      strings, booleans, etc.
    - Export/load functions can be specified according to file system.

'''
import os
from logging import getLogger
from beiwetools.helpers.functions import write_json, read_json


logger = getLogger(__name__)


class SerializableClass():
    '''
    Attributes:
        do_not_serialize (list): List of attributes (str) that should not be 
            serialized.
                
    '''
    def __init__(self):
        '''
        Leave empty.  Use create() or load() to initialize an instance.
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
        
    def export(self, name, out_dir, 
               export_function = write_json, **export_kwargs):
        '''
        Serialize object attributes and write to a JSON file.
        
        Args:
            name (str): Name for the JSON file, without extension.
            out_dir (str): Full path to directory where JSON file will be 
                written.
            export_function (function): A function that writes dictionaries to
                JSON files.  Specifications:
                - Args:
                    dictionary (dict): Dictionary to write to JSON file.
                    name (str):  Name for the file to create, without extension.
                    dirpath (str):  Path to location for the JSON file.
                    export_kwargs (dict): Other keyword arguments.                       
                - Returns:
                    filepath (str): Path to the JSON file.
            **export_kwargs: Keyword arguments for export_function.
                
        Returns:
            filepath (str): Path to JSON with serialized attributes.
        '''
        try:
            all_attributes = self.__dict__.keys()
            select_attributes = [k for k in all_attributes if not k in self.do_not_serialize]
            dict_to_write = {k:self.__dict__[k] for k in select_attributes}
            filepath = 

            class_file = os.path.join(out_dir, name + '.json')
            with open(class_file, 'w') as f:
                json.dump(dict_to_write, f, indent = indent)


            return(filepath)
        except:
                logger.warning('Unable to write JSON file.')
    
    @classmethod
    def load(cls, filepath, 
             loader = read_json, **load_kwargs):
        '''
        Given a file with serialized object attributes, initialize a new class
        instance with those attributes.

        Args:
            filepath (str): Full path to a file with serialized 
                object attributes.
            loader (function):
                
                
                
            **load_kwargs: Keyword arguments for loader.
            
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
        Define equivalence.
        '''
        same = False
        if type(other) == type(self):        
            try:
                same = other.__dict__ == self.__dict__
            except:
                logger.warning('Equivalence is not implemented.')                
        return(same)