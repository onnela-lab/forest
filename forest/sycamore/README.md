### Authors: Nellie Ponarul, Anna Beukenhorst  

### Last Update Date: 24 February 2021  

### Executive Summary: 
Use `sycamore` to process and analyze Beiwe survey data.


#### Installation

`from forest import sycamore.functions as syc`

## Usage:  
Download raw data from your Beiwe server and use this package to process the data in the `survey_timings` data stream.   

## Data:   
Methods are designed for use on the `survey_timings` data from the Beiwe app.

___
## Functions  
1.  [`syc.parse_timings`](#config)
2.  [`syc.aggregate_surveys_config`](#agg)  
3.  [`syc.get_survey_timings`](#get)

___
## 1. `syc.parse_timings` <a name = "config"/>  
Takes the path to the study configuration file and outputs a table summarizing the questions and scheduled timings of each survey.  

*Example*  
```
config_path = path/to/config file
surveys_info = syc.parse_timings(config_path)
```
___
## 2. `syc.aggregate_surveys_config` <a name="agg"/>
Takes a path to raw data and returns aggregated `survey_timings` data with timings information from the study configuration file in a tabular format. This is useful for analysis in Tableau or other formats. Has the option to add fields that will calculate the time lapse between the expectation survey notification and line in the survey data. If this option is used, the study timzone must also be supplied through the `study_tz` argument.  

*Example*  
```
path = 'path/to/data'  
config_path = 'path/to/study_config_file'
study_tz = 'America/New_York'
# Without time lapse fields
survey_data = syc.aggregate_surveys_config(path, config)
# With time lapse fields
survey_data = syc.aggregate_surveys_config(path, config, calc_time_diff = True, study_tz = study_tz)
```
  
___
## 3. `syc.get_survey_timings` <a name="get"/>  
Extracts the beginning and submission times for each survey instance in a given study and survey (using the survey ID), using the `survey_timings` data.  

*Example*  
```
SURVEY_ID = survey_id
RAW_DATA_DIR_HERE = path/to/data
all_ptcp = os.listdir(path)
PATH_TO_OUTPUT_FILE = path/to/ouput

survey_timings_array = get_survey_timings(all_ptcp,
                   RAW_DATA_DIR_HERE,
                   SURVEY_ID)
                   
survey_timings_array.to_csv(PATH_TO_OUTPUT_FILE.csv, index = False)
```
