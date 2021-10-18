### Authors: Nellie Ponarul, Anna Beukenhorst  

### Last Update Date: 8 April 2021  

### Executive Summary: 
Use `sycamore` to process and analyze Beiwe survey data.


#### Installation

`from forest import sycamore.sycamore_main as syc`  

`from forest import sycamore.functions as sycf`

## Usage:  
Download raw data from your Beiwe server and use this package to process the data in the `survey_timings` data stream. Summary data provides metrics around survey submissions and survey question completion. Additional outputs are generated if a config file is provided.

If config file is not provided, outputs include:  

* `agg_survey_data.csv`: One csv with all data from the survey_timings datastresm.
* `submits_alt_summary.csv`: One csv with survey completion times for all users.  

If a config file is provided, additional outputs include:

* `answers_data.csv`: Aggregated survey data (similar to agg_survey_data.csv, will likely be deprecated in future iterations).
* `answers_summary.csv` : For each beiwe id, survey id, and question id, this file provides the following summaries:
  * `num_answers`: The number of times in the given data the answer is answered.
  * `average_time_to_answer`: The average amount of time the user takes to answer the question.
  * `average_number_of_answers`: Average number of answers selected for a question. This indicated if a user changed an answer before submitting it.  
  * `most_common_answer`: A user's most common answer to a question.


* `submits_summary.csv`: For each survey id and beiwe id, this file provides the following summaries:  
  * `num_surveys`: The number of surveys issued to the user in the given timeframe. This is calculated from the provided config file.
  * `num_completed_surveys`: The number of surveys the user submitted in the given timeframe. A completed survey is considered a survey that was completed before the next survey was deployed to the user.
  * `avg_time_to_submit`: The average amount of time that lapses between the survey getting deployed and the user submitting the survey.  
* `submits_data.csv`: For each survey id and beiwe id, this file provides survey timings details summarized in `submits_summary.csv`

## Data:   
Methods are designed for use on the `survey_timings` data from the Beiwe app.

___
## Functions  
1.  [`syc.sycamore_main`](#syc_main)
2.  [`sycf.get_survey_timings`](#get)

___
## 1. `syc.sycamore_main` <a name = "syc_main"/>  

*Example (without config file)*    
```
study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
time_start = start time
time_end = end time  
study_tz = Timezone of study (if not defined, defaults to 'America/New_York')

sycamore_main.survey_stats_main(output_dir, study_dir, beiwe_ids, time_start = time_start, time_end = time_end, study_tz)
```

```
config_path = path/to/config file
study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
time_start = start time
time_end = end time  
study_tz = Timezone of study (if not defined, defaults to 'America/New_York')

sycamore_main.survey_stats_main(output_dir, study_dir, config_path, time_start, time_end, beiwe_ids, study_tz)

```
___
## 2. `sycf.get_survey_timings` <a name="get"/>  
Extracts the beginning and submission times for each survey instance in a given study and survey (using the survey ID), using the `survey_timings` data.  

*Example*  
```
SURVEY_ID = survey_id
RAW_DATA_DIR_HERE = path/to/data
all_ptcp = os.listdir(path)
PATH_TO_OUTPUT_FILE = path/to/ouput

survey_timings_array = sycf.get_survey_timings(all_ptcp,
                   RAW_DATA_DIR_HERE,
                   SURVEY_ID)
                   
survey_timings_array.to_csv(PATH_TO_OUTPUT_FILE.csv, index = False)
```
