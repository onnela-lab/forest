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

## Data:   
Methods are designed for use on the `survey_timings` and `survey_answers` data from the Beiwe app.

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
