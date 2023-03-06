# Sycamore

## Executive Summary: 

Use `sycamore` to process and analyze Beiwe survey data.

## Installation

Before using sycamore, dependencies for librosa (ffmpeg and libsndfile1) must be installed first in order to enable processing of audio survey files.  

To install these dependencies on ubuntu, simply run:  
`sudo apt-get install -y ffmpeg libsndfile1`  

For more information, see the [librosa documentation](https://librosa.org/doc/latest/install.html)

## Import

User-facing functions can be imported directly from sycamore:

`from forest.sycamore import compute_survey_stats`  
`from forest.sycamore import aggregate_surveys_config` 
`from forest.sycamore import survey_submits` 
`from forest.sycamore import survey_submits_no_config` 
`from forest.sycamore import agg_changed_answers_summary` 

## Usage:   
Download raw data from your Beiwe server and use this package to process the data in the `survey_timings`, `survey_answers`, and `audio_recordings` data streams, using `survey_answers` as a backup for possible missing `survey_timings` files. Summary data provides metrics around survey submissions and survey question completion. Sycamore takes various auxiliary files which can be downloaded from the Beiwe website to ensure accurate output.  

## Data:   
Methods are designed for use on the `survey_timings` and `survey_answers` data from the Beiwe app.

## Auxiliary files:   
Sycamore requires users to manually download files from the Beiwe website to create some outputs. These files can be downloaded by clicking "Edit this Study" on the study page, and clicking on the relevant file. When running Sycamore, pass the path to the file downloaded by clicking "Export study settings JSON file" under "Export/Import study settings" to the `config_path` argument. Pass the file downloaded by clicking "Download Interventions" next to "Intervention Data" to the `interventions_filepath` argument. And, pass the file downloaded by clicking "Download Surveys" next to "Survey History" to the `history_path` argument. 

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
start_date = "2022-01-01"
end_date = "2022-06-04"
study_tz = Timezone of study (if not defined, defaults to 'UTC')

compute_survey_stats(
    study_dir, output_dir, study_tz, beiwe_ids, start_date=start_date, 
    end_date=end_date
)
```

*Example (with config file)* 
```
config_path = path/to/config file
interventions_filepath = path/to/interventions file
history_path = path/to/history/file
study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
start_date = "2022-01-01"
end_date = "2022-06-04"
study_tz = Timezone of study (if not defined, defaults to 'UTC')


compute_survey_stats(
    study_dir, output_dir, study_tz, beiwe_ids, start_date=start_date, 
    end_date=end_date, config_path, interventions_filepath, 
    history_path=history_path
)

```

Most users should be able to use `compute_survey_stats` for all of their survey processing needs. However, if a study has collected a very large number of surveys, subprocesses are also exposed to reduce processing time. 

___
## 2. `sycamore.common.aggregate_surveys_config`

Aggregate all survey information from a study, using the config file to infer information about surveys

*Example*  
```
from forest.sycamore import aggregate_surveys_config

agg_data = aggregate_surveys_config(study_dir, config_path, study_tz, history_path=history_path)
```

___
## 3. `sycamore.submits.survey_submits` 

Extract and summarize delivery and submission times

*Example*  
```
from forest.sycamore.submits import survey_submits

config_path = path/to/config file
interventions_path = path/to/interventions file
history_path = path/to/history/file
study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
time_start = start time (the first week the script will assume weekly surveys are delivered)
time_end = end time  (the last week the script will assume weekly surveys are delivered)
study_tz = Timezone of study (if not defined, defaults to 'UTC')

agg_data = aggregate_surveys_config(study_dir, config_path, study_tz)

submits_detail, submits_summary = survey_submits(
    config_path, time_start, time_end, beiwe_ids, interventions_path, agg_data, 
    history_path
)
```
 
___
## 4. `sycamore.submits.survey_submits_no_config`
Used to extract an alternative survey submits table that does not include delivery times

*Example*  
```
from forest.sycamore import survey_submits_no_config

study_dir = path/to/data  

submits_tbl = survey_submits_no_config(study_dir)

```
 
___
## 5. `sycamore.responses.agg_changed_answers_summary`
Used to extract data summarizing user responses
 
*Example*  
```
from forest.sycamore import agg_changed_answers_summary

config_path = path/to/config file
history_path = path/to/history/file
study_dir = path/to/data  
output_dir = path/to/output
beiwe_ids = list of ids in study_dir
time_start = start time
time_end = end time  
study_tz = Timezone of study (if not defined, defaults to 'UTC')

agg_data = aggregate_surveys_config(study_dir, config_path, study_tz, history_path=history_path)

ca_detail, ca_summary = agg_changed_answers_summary(config_path, agg_data)
 
```
