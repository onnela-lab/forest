import os
import functions as agg
import survey_config as sc
import changed_answers as ca

def survey_stats_main(output_dir, study_dir, config_path, time_start, time_end, beiwe_ids, study_tz = None): 
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
    agg_data = agg.aggregate_surveys_config(study_dir, config_path, study_tz)
    
    # Create changed answers detail and summary
    ca_detail, ca_summary = ca.agg_changed_answers_summary(study_dir, config_path, agg_data, study_tz)
    
    # Create survey submits detail and summary
    ss_detail, ss_summary = sc.survey_submits(study_dir, config_path, time_start, time_end, beiwe_ids, agg_data, study_tz)
    
    # Write out summaries
#     agg_data.to_csv(os.path.join(output_dir, 'agg_survey_data.csv'), index = False)
    ca_detail.to_csv(os.path.join(output_dir, 'answers_data.csv'), index = False)
    ca_summary.to_csv(os.path.join(output_dir, 'answers_summary.csv'), index = False)
    ss_detail.to_csv(os.path.join(output_dir, 'submits_data.csv'), index = False)
    ss_summary.to_csv(os.path.join(output_dir, 'submits_summary.csv'), index = False)
    
    
    return 