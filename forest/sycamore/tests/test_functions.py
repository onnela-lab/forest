import pandas as pd

from forest.sycamore.survey_config import get_all_interventions_dict
from forest.sycamore.functions import aggregate_surveys
from forest.sycamore.survey_config import gen_survey_schedule

assert get_all_interventions_dict("empty_intervention_data.json") == {}

interventions_dict = get_all_interventions_dict(
    "sample_intervention_data.json")

assert type(interventions_dict["ntz4apjf"]) is dict

assert interventions_dict["ntz4apjf"]["Time_0"] == "2021-12-14"


sample_agg_data = aggregate_surveys("sample_dir", ["idr8gqdh"])


pd.isnull(sample_agg_data.loc[0, "time_prev"])

assert all(pd.to_datetime(sample_agg_data['timestamp'], unit='ms') !=
           sample_agg_data['UTC time'])


sample_schedule = gen_survey_schedule(
    "sample_study_surveys_and_settings.json",
    time_start=pd.to_datetime("2021-12-01"),
    time_end=pd.to_datetime("2021-12-30"),
    beiwe_ids=["idr8gqdh"],
    all_interventions_dict=interventions_dict)

assert sample_schedule.shape[0] == 7

assert all(
    sample_schedule.columns == pd.Index(['delivery_time',
                                         'next_delivery_time', 'id',
                                         'beiwe_id', 'question_id']))
