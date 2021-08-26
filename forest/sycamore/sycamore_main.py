import os
from .functions import aggregate_surveys_config, aggregate_surveys_no_config
from .survey_config import survey_submits, survey_submits_no_config
from .changed_answers import agg_changed_answers_summary

def survey_stats_main(output_dir, study_dir, beiwe_ids, config_path = None, time_start = None, time_end = None, study_tz = None): 
    '''
    Args:
    output_dir(str):
        File path to output summaries and details
    study_dir(str):
        File path to study data
    config_path(str):
        File path to study configuration file
    time_start(str):
        The first date of the survey data
    time_end(str):
        The last date of the survey data
    beiwe_ids(list):
        List of users in study for which we are generating a survey schedule        
    study_tz(str):
        Timezone of study. This defaults to 'America/New_York'
    '''
    
    # Read, aggregate and clean data 
    if config_path is None:
        print('No config file provided. Skipping some summary outputs.')
        agg_data = aggregate_surveys_no_config(study_dir, study_tz)

    else:
        agg_data = aggregate_surveys_config(study_dir, config_path, study_tz)
         # Create changed answers detail and summary
        ca_detail, ca_summary = agg_changed_answers_summary(study_dir, config_path, agg_data, study_tz)
        if time_start is not None and time_end is not None:
            # Create survey submits detail and summary
            ss_detail, ss_summary = survey_submits(study_dir, config_path, time_start, time_end, beiwe_ids, agg_data, study_tz)
                        # Write out summaries
        #     agg_data.to_csv(os.path.join(output_dir, 'agg_survey_data.csv'), index = False)
            ca_detail.to_csv(os.path.join(output_dir, 'answers_data.csv'), index = False)
            ca_summary.to_csv(os.path.join(output_dir, 'answers_summary.csv'), index = False)
            ss_detail.to_csv(os.path.join(output_dir, 'submits_data.csv'), index = False)
            ss_summary.to_csv(os.path.join(output_dir, 'submits_summary.csv'), index = False)
    
    # Add alternative survey submits table
    submits_tbl = survey_submits_no_config(study_dir, study_tz)
    submits_tbl.to_csv(os.path.join(output_dir, 'submits_alt_summary.csv'), index = False)
    
    
    return 