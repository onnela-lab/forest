# Sycamore

## Executive Summary: 

Use `sycamore` to process and analyze Beiwe survey data.

## Import

User-facing functions can be imported directly from sycamore:

`from forest.sycamore import compute_survey_stats`  
`from forest.sycamore import aggregate_surveys_config` 
`from forest.sycamore import survey_submits` 
`from forest.sycamore import survey_submits_no_config` 
`from forest.sycamore import agg_changed_answers_summary` 

## Usage:  
Download raw data from your Beiwe server and use this package to process the data in the `survey_timings` data stream, using `survey_answers` as a backup for possible missing `survey_timings` files. Summary data provides metrics around survey submissions and survey question completion. Additional outputs are generated if a config file is provided.

If config file is not provided, outputs include:  

### Results of sycamore.common.aggregate_surveys_no_config

* `agg_survey_data`: One csv/dataframe with all data from the survey_timings data stream.
  * `timestamp`: The Unix timestamp corresponding to the event
  * `UTC time`: The time (in UTC time) corresponding to the event
  * `question id`: The ID of the question corresponding to the event
  * `survey id`: The ID of the survey corresponding to the event
  * `question type`: The type of question (radio button, free response, etc.) corresponding to the event
  * `question text`: The text presented to the user
  * `question answer options`: The answer options presented to the user (applicable for check box or radio button surveys)
  * `answer`: The answer selected by the user at the time corresponding to the event. 
  * `event`: The event (the user may have changed the answer, or they may have pushed submit, for example)
  * `beiwe_id`: The Beiwe ID of the user corresponding to the event
  * `DOW`: The day of the week corresponding to the event.
  * `question index`: An integer corresponding to the order in which questions are answered. If an individual on question 3 goes to question 4 and then back to question 3, the individual will have two distinct `question_index` values for question 3. 
  * `surv_inst_flg`: A number used to distinguish different administrations of the same user for an individual
  * `time_prev`: The UTC time corresponding to the previous event
  * `time_diff`: The time difference between the current event and the previous event
  * `config_id`: A number corresponding to the survey_id
  * `submit_line`: Whether an individual's time includes a submission. 
  * `Local time`: The time of the event, in local time
  * `last_answer`: The last answer to a question included

### Results of sycamore.submits.survey_submits_no_config

* `submits_alt_summary`: One csv/dataframe with survey completion times for all users.  
  * `survey id`: The ID of the survey corresponding to the submission
  * `beiwe_id`: The Beiwe ID of the user corresponding to the submission
  * `surv_inst_flg`: A number used to distinguish different administrations of the same user for an individual
  * `max_time` The latest time in which the participant was in the survey, in local time
  * `min_time` The earliest time in which the participant was in the survey, in local time
  * `time_to_complete` The duration of the survey, in seconds.

If a config file is provided, additional outputs include:

### Results of sycamore.responses.agg_changed_answers_summary

* `answers_data`: Aggregated survey data (similar to agg_survey_data, will likely be deprecated in future iterations).
  * `survey id`: The survey ID corresponding to the answer
  * `beiwe_id`: The Beiwe ID of the user corresponding to the answer
  * `question id`: The question ID corresponding to the answer
  * `question text`: The question text corresponding to the answer
  * `question type`: The type of question (radio button, free response, etc.) corresponding to the answer
  * `question answer options`: The answer options presented to the user (applicable for check box or radio button surveys)
  * `timestamp`: The Unix timestamp corresponding to the latest time the user was on the question
  * `Local time`: The local time corresponding to the latest time the user was on the question
  * `last_answer`: The last answer the user had selected before moving on to the next question or submitting
  * `all_answers`: A list of all answers the user selected
  * `num_answers`: The number of different answers selected by the user (the length of the list in all_answers)
  * `first_time`: The local time corresponding to the earliest time the user was on the question
  * `last_time`: The local time corresponding to the latest time the user was on the question
  * `time_to_answer`: The time that the user spent on the question

* `answers_summary` : For each beiwe id, survey id, and question id, this file/dataframe provides the following summaries:
  * `num_answers`: The number of times in the given data the answer is answered.
  * `average_time_to_answer`: The average amount of time the user takes to answer the question.
  * `average_number_of_answers`: Average number of answers selected for a question. This indicated if a user changed an answer before submitting it.  
  * `most_common_answer`: A user's most common answer to a question.

### Results of sycamore.submits.survey_submits

* `submits_summary`: For each survey id and beiwe id, this file/dataframe provides the following summaries:  
  * `num_surveys`: The number of surveys issued to the user in the given timeframe. This is calculated from the provided config file.
  * `num_completed_surveys`: The number of surveys the user submitted in the given timeframe. A completed survey is considered a survey that was completed before the next survey was deployed to the user.
  * `avg_time_to_submit`: The average amount of time that lapses between the survey getting deployed and the user submitting the survey.  

* `submits_data` : For each survey id and beiwe id, this file/dataframe provides survey timings details summarized in `submits_summary`
  * `survey_id`: The survey ID corresponding to this delivery
  * `delivery_time`: The time the survey was scheduled to be delivered 
  * `beiwe_id`: The participant's Beiwe ID
  * `submit_flg`: An indicator corresponding to whether the user submitted a survey between this delivery time and the next delivery time
  * `submit_time`: The submission time of the survey, if applicable
  * `time_to_submit`: The difference between submission time and delivery time of the survey, if applicable


## Data:   
Methods are designed for use on the `survey_timings` data from the Beiwe app.

___
## Functions  
1.  [`sycamore.base.compute_survey_stats`](#1-sycamorebasecompute_survey_stats)
2.  [`sycamore.common.aggregate_surveys_config`](#2-sycamorecommonaggregate_surveys_config)
3.  [`sycamore.submits.survey_submits`](#3-sycamoresubmitssurvey_submits)
4.  [`sycamore.submits.survey_submits_no_config`](#4-sycamoresubmitssurvey_submits_no_config)
5.  [`sycamore.responses.agg_changed_answers_summary`](#5-sycamoreresponsesagg_changed_answers_summary)

___
## 1. `sycamore.base.compute_survey_stats` 

compute_survey_stats runs aggregate_surveys_config, survey_submits, survey_submits_no_config, and agg_changed_answers_summary, and writes their output to csv files


*Example (without config file)*    
```
from forest.sycamore import compute_survey_stats

study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
time_start = start time
time_end = end time  
study_tz = Timezone of study (if not defined, defaults to 'UTC')

compute_survey_stats(
    study_dir, output_dir, study_tz, beiwe_ids, time_start=time_start, 
    time_end = time_end
)
```

*Example (with config file)* 
```
config_path = path/to/config file
interventions_path = path/to/interventions file
study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
time_start = start time
time_end = end time  
study_tz = Timezone of study (if not defined, defaults to 'UTC')

compute_survey_stats(
    study_dir, output_dir, study_tz, beiwe_ids, time_start=time_start, 
    time_end=time_end, config_path, interventions_path
)

```

___
## 2. `sycamore.common.aggregate_surveys_config`

Aggregate all survey information from a study, using the config file to infer information about surveys

*Example*  
```
from forest.sycamore import aggregate_surveys_config

agg_data = aggregate_surveys_config(study_dir, config_path, study_tz)
```

___
## 3. `sycamore.submits.survey_submits` 

Extract and summarize delivery and submission times

*Example*  
```
from forest.sycamore.submits import survey_submits

config_path = path/to/config file
interventions_path = path/to/interventions file
study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
time_start = start time (the first week the script will assume weekly surveys are delivered)
time_end = end time  (the last week the script will assume weekly surveys are delivered)
study_tz = Timezone of study (if not defined, defaults to 'UTC')

agg_data = aggregate_surveys_config(study_dir, config_path, study_tz)

all_interventions_dict = get_all_interventions_dict(interventions_path)

submits_detail, submits_summary = survey_submits(
    config_path, time_start, time_end, beiwe_ids, agg_data, 
    all_interventions_dict
)
```
 
___
## 4. `sycamore.submits.survey_submits_no_config`
Used to extract an alternative survey submits table that does not include delivery times

*Example*  
```
from forest.sycamore import survey_submits_no_config

study_dir = path/to/data  
study_tz = Timezone of study (if not defined, defaults to 'UTC')

submits_tbl = survey_submits_no_config(study_dir, study_tz)

```
 
___
## 5. `sycamore.responses.agg_changed_answers_summary`
Used to extract data summarizing user responses
 
*Example*  
```
from forest.sycamore import agg_changed_answers_summary

config_path = path/to/config file
study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
time_start = start time
time_end = end time  
study_tz = Timezone of study (if not defined, defaults to 'UTC')

agg_data = aggregate_surveys_config(study_dir, config_path, study_tz)

ca_detail, ca_summary = agg_changed_answers_summary(config_path, agg_data)
 
```
