'''Miscellaneous classes for working with Beiwe data and study configurations.

'''
import os
import logging
import datetime
import textwrap
import numpy as np
import pandas as pd
from collections import OrderedDict
from .time import local_time_format
from .functions import (write_string, setup_directories, 
                        setup_csv, write_to_csv, check_same)


logger = logging.getLogger(__name__)


class Timer():
    '''
    Simple class for timing data processing events.    
    
    Args:
        unit (str): Return elpased time in this unit.
            'seconds', 'minutes', or 'hours'
        ndigits (int): Number of digits for rounding.
        to_format (str): Directives for returning local time.
        
    Attributes:
        ready (bool):  True if timer is not timing anything.
        unit (str): Return elpased time in this unit.
            'seconds', 'minutes', or 'hours'
        ndigits (int): Number of digits for rounding.
        to_format (str): Directives for returning local time.
        divide_by (int):  Integer to divide by to return correct units.
        t0 (NoneType or int): 
            None if nothing is being timed. Otherwise, a timestamp in seconds.        
    '''
    def __init__(self, unit = 'minutes', ndigits = 2,
                  to_format = local_time_format):
        self.ready = True
        self.unit = unit
        self.ndigits = ndigits
        if not '%Z' in to_format:
            logging.warning('Time format doesn\'t include time zone.')
        self.to_format = to_format
        self.t0 = None
        if   self.unit == 'seconds': self.divide_by = 1
        elif self.unit == 'minutes': self.divide_by = 60
        elif self.unit == 'hours':   self.divide_by = 60*60  

    def start(self):
        if self.ready:
            self.ready = False
            now = datetime.datetime.now()
            self.t0 = now.timestamp()
            local = now.astimezone().strftime(self.to_format)
            return(local)
        else:
            logging.warning('Timer is already timing something.')
        
    def stop(self):
        if not self.ready:
            self.ready = True
            now = datetime.datetime.now()
            t1 = now.timestamp()
            local = now.astimezone().strftime(self.to_format)
            elapsed = round((t1 - self.t0)/self.divide_by, self.ndigits)
            self.t0 = None
            return(local, elapsed, self.unit)
        else:
            logging.warning('Timer wasn\'t timing anything.')


class Summary():
    '''
    Class for generating printable summaries.

    Args:
        labels (list):  Corresponding labels (str) for items in the list of values.
        items (list):  A list of items to be labeled.
            Items can be NoneType, int, bool, str, OrderedDict, Summary.
            An item can also be a list, but only containing NoneType, int, bool, str.
        header (str): An optional header to print.
        sep (str):  
            How to separate labels from values when printing.  Usually ': '. 

    Attributes:
        contents (OrderedDict):
            Keys (str) are labels for values.
            Values are None, integers, booleans, strings, OrderedDicts or Summary objects.
        Other attributes: See Args.
    '''
    def __init__(self, labels, items, header = None, sep = ': '):
        if len(labels) > len(items):
            logging.warning('Too many labels.')
        if len(labels) < len(items):
            logging.warning('Not enough labels.')
        self.contents = OrderedDict(zip(labels, items))
        self.header = header
        self.sep = sep
        
    def __eq__(self, other):
        return(check_same(self, other))
        
    def to_lines(self, level = 0, rule_zero = True, extra_breaks = [],
                 indent = 4, max_width = 70):
        '''
        Convert summary contents to lines that can be formatted for printing.
        
        Args:    
            level (int):  Top level of the Summary object, for determining indentation.
            rule_zero (bool):  
                If True, horizontal rules are added above and below the top level labels.
                Use especially when the summary contains a lot of nested items.
            extra_breaks (list):  
                An extra line break is added before labels on levels in this list.
                Use when contents contain a lot of paragraphs.
            indent (int):  Number of spaces for indenting.
            max_width (int):  Maximum number of characters per line.
            
        Returns:
            lines (list):  A list of pairs (indentation, text).  
                Indentation (int) is the number of spaces to indent the text (str).
        ''' 
        sep = self.sep
        do_rules = False
        do_break = False        
        if level == 0 and rule_zero:
            do_rules = True
            rule_pre = '\n' + '-' * max_width
            rule_post = '-' * max_width + '\n'
        if level in extra_breaks:
            do_break = True
        lines = []        
        for k in self.contents.keys():
            if do_break:  lines.append((0, '\n'))
            c = self.contents[k]
            # format nonetype
            if c is None: c = 'Not found'
            # format list
            if type(c) is list:
                c = str(c).replace("'", "")[1:-1]
            # handle int, bool, str
            if type(c) in [int, bool, str, float]:
                c = str(c)
                if do_rules or len(c) > max_width - len(k) - len(sep) - (level*indent):
                    if do_rules:
                        lines += [(0, rule_pre),(0,  k), (0, rule_post)]
                    else:
                        lines.append((level*indent, k + sep))
                    lines.append(((level+1)*indent, c))
                else:
                    lines.append((level*indent, k + sep + c))
            # handle OrderedDict or Summary object
            else:  
                if type(c) is OrderedDict:
                    c = Summary([e for e, a in c.items()], [a for e, a in c.items()])
                if do_rules:
                    lines += [(0, rule_pre),(0,  k), (0, rule_post)]
                else:
                    lines.append((level*indent, k + sep))
                lines += c.to_lines(level = level+1, extra_breaks = extra_breaks,
                                    indent = indent, max_width = max_width)
        return(lines)
                
    def to_string(self, rule_zero = True, extra_breaks = [],
                  indent = 4, max_width = 70):   
        '''
        Convert summary contents to a formatted string for printing.

        Args:
            See to_lines.
            
        Returns:
            readable (str): A string formatted for printing.        
        '''
        lines = self.to_lines(level = 0, rule_zero = rule_zero, extra_breaks = extra_breaks,
                              indent = indent, max_width = max_width)
        formatted_lines = ['']
        for i in range(len(lines)):
            indentation = lines[i][0] * ' '
            for j in lines[i][1].splitlines():
                temp = textwrap.fill(j, width = max_width,
                                     initial_indent = indentation, 
                                     subsequent_indent = indentation) 
                formatted_lines.append(temp)
        formatted_lines.append('')
        readable = '\n'.join(formatted_lines)
        if not self.header is None:
            readable = '\n' + self.header + readable
        return(readable)
        
    def print(self, rule_zero = True, extra_breaks = [],
              indent = 4, max_width = 70):
        '''
        Print summary.

        Args:
            See to_lines.
            
        Returns:
            None
        '''
        readable = self.to_string(rule_zero = rule_zero, 
                                  extra_breaks = extra_breaks,
                                  indent = indent, max_width = max_width)        
        print(readable)

    def to_file(self, name, directory, mode = 'w', 
                rule_zero = True, extra_breaks = [],
                indent = 4, max_width = 70):
        '''
        Export summary to a text file.
        
        Args:
            name (str):  Name of the text file.
            directory (str):  Path to location to write the text file.
            mode (str):  Mode for open().  Use 'w' to overwrite, 'a' to append.
            Other arguments:  See to_lines.          

        Returns:
            None
        '''
        readable = self.to_string(rule_zero = rule_zero, 
                                  extra_breaks = extra_breaks,
                                  indent = indent, max_width = max_width)
        write_string(readable, name, directory, mode = mode)        

    
class Transformation():
    '''
    Class to represent transformations of variables in a data frame.
    
    Args:
        name (str): Human-readable name for the transformation, e.g. for use in csv headers.
        long_name (str): A long name for use in plots etc.
        unit (str): Unit of transformed data.
        data_type (str):  A Beiwe data stream, e.g. 'accelerometer' or 'gps'.
        f (function): Should take a data frame with a raw Beiwe data header.
            May require additional columns to be present.    
            May take additional arguments.
            Should return one or more numpy arrays with as many entries as rows in the input data frame.
        returns (list): List of names for f's output, for use in csv headers.
            If None, then f should return just one numpy array.
            Otherwise, returns should contain a name for each array in f's output.
                            
    Attributes:
        Same as Args.    
    '''
    def __init__(self, name, long_name, unit, data_type, f, returns = None):
        self.name = name
        self.long_name = long_name
        self.unit = unit
        self.data_type = data_type
        self.f = f
        if returns is None:
            self.returns = [self.name]
        else:
            self.returns = returns
        
    def apply(self, df, **kwargs):
        '''
        Calculates the transformation for each row and appends to df.

        Args:
            df (DataFrame): A pandas dataframe obtained from a raw Beiwe data file.
                May contain some additional columns.
            **kwargs:  Additional arguments for f, if any.   
        
        Returns:
            None
        '''
        for r in self.returns:
            df[r] = np.nan
        df[self.returns] = self.f(df, **kwargs)


class WriteQueue():
    '''
    Manager for writing line-by-line output to multiple CSVs.
    Use this class to avoid creating arbitrarily large files.
    
    Args:
        name (str): Name for folder, prefix for CSVs.
        directory (str): Where to write new folder.
        header (list): List of column headers.
        new_folder (bool): Whether to create a new folder for CSVs.
        lines_per_csv (int): Maximum number of lines per CSV.
        n_anticipated_files (int or Nonetype): 
            Optionally, provide an estimate the number of files that will be written.
            If None, will assume NTFS maximum of around 10^10.
            Better to overestimate, or just leave as None.
        
    Attributes:
        name (str): Name for folder, prefix for CSVs.
        directory (str): Where to write CSVs.
        header (list): List of column headers.
        lines_per_csv (int): Maximum number of lines per CSV.
        n_digits (int): Length of file identifier suffix.
        line_count (int): Number of lines written to current file.
        file_count (int): Number of files written.
        file_list (list): Paths to files that have been created.
    '''
    def __init__(self, name, directory, header, new_folder = True,
                 lines_per_csv = 100, n_anticipated_files = None):        
        self.name = name
        self.header = header
        self.lines_per_csv = lines_per_csv
        if new_folder:
            # create directory
            self.directory = os.path.join(directory, name)
            setup_directories(self.directory)
        else:
            self.directory = directory
        if n_anticipated_files is None:
            self.n_digits = 10
        else:
            self.n_digits = len(str(n_anticipated_files))
        # set up counts and path records
        self.line_count = 0
        self.file_count = 0
        self.current_path = None 
        self.file_list = []
        # initialize the first CSV
        self.new_file()
    
    def new_file(self):
        '''
        Prepare a new CSV.
        '''
        n = str(self.file_count).zfill(self.n_digits) 
        new_name = '%s_%s' % (self.name, n)
        self.current_path = setup_csv(new_name, self.directory, self.header)        
        self.file_list.append(new_name)
        self.file_count += 1
        self.line_count = 0
        
    def write(self, line):
        '''
        Write a line to a CSV.
        '''
        if len(line) != len(self.header):
            logging.warning('The line doesn\'t match the number of columns.')
        if self.line_count == self.lines_per_csv:
            self.new_file()
        write_to_csv(self.current_path, line)
        self.line_count += 1
        
    
class ReadQueue():
    '''
    Manager for reading multiple CSVs that contain contiguous data.  Delivers CSVs in chunks.
    Use this class to avoid reading arbitrarily many CSVs at a time, 
    e.g. when a data processing task requires reading two or more contiguous files at a time.

    Args:
        filepaths (list): List of paths to files, in the order in which they should be read.
        return_as (str): 
            If "lines", delivers each chunk as a list of lines.
            If "dataframe", delivers each chunk as a pandas dataframe.
        ignore_header (bool): 
            Only matters if return_as == "lines".
            Ignore the first line of each CSV.
        chunk_size (int): How many files to deliver at a time.
        
    Attributes:
        Same as Args.        
    '''
    def __init__(self, filepaths, return_as = 'dataframe', 
                 ignore_header = True, chunk_size = 1):
        self.filepaths = filepaths
        self.return_as = return_as
        self.ignore_header = ignore_header
        self.chunk_size = chunk_size        
                
    def get(self):
        '''
        Return the next chunk of files.
        '''
        if len(self.filepaths) > self.chunk_size:
            temp = self.filepaths[0:self.chunk_size]
            self.filepaths = self.filepaths[self.chunk_size:]
        else:
            temp = self.filepaths
            self.filepaths = []
        if len(temp) > 0:
            if self.return_as == 'lines':
                lines = []
                for p in temp:
                    f = open(p, 'r')
                    new_lines = list(f)
                    if self.ignore_header: new_lines = new_lines[1:]
                    lines += new_lines
                    f.close()
                # drop line breaks
                for i in range(len(lines)):
                    lines[i] = lines[i].replace('\n', '')
                return(lines)
            elif self.return_as == 'dataframe':
                dataframes = []
                for p in temp:
                    dataframes.append(pd.read_csv(p))
                return(pd.concat(dataframes, ignore_index = True))
        else: return(None)
