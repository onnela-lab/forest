### Authors: Nellie Ponarul, Anna Beukenhorst  

### Last Update Date: 3 February 2021  

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
1.  [`syc.aggregate_survey_timings`](#agg)  
2.  [`syc.get_survey_timings`](#get)

___
## 1. `syc.aggregate_survey_timings` <a name="agg"/>
Takes a path to raw data and returns aggregated `survey_timings` data in separate tabular datasets. This is useful for processing data for use in another platform like Tableau:  

1. **Questions dataset**: Contains the questions and answers for all survey instances (an instance is single survey completion by a user for a survey) in a study.
2. **Starts dataset**: Contains all of the survey start times for all instances
3. **Submits dataset**: Contains all of the survey end times for all instances
4. **Survey notifications** (iOS): Contains all of the survey notifications data for all instances, like the time the user is notified a survey has been delievered, and if the user did not take the survey, the time the notification expired. Note, this data is only collected for iOS users.
5. **Survey question times** (iOS): Contains all metadata on when a question is presented to a user and unpresented to a user across all survey instances. Note, this data is only collected for iOS users.  

*Example*  
```
path = 'path/to/data'  
questions, starts, submits, notif, q_times = syc.aggregate_survey_timings(path)
```
  
___
## 2. `syc.get_survey_timings` <a name="get"/>  
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
