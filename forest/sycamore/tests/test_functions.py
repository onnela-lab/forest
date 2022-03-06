import numpy as np
import os
import pandas as pd


from forest.sycamore.survey_config import get_all_interventions_dict
from forest.sycamore.functions import aggregate_surveys
from forest.sycamore.survey_config import gen_survey_schedule

# Needed so that relative paths in the below file will work
os.chdir(os.path.join("forest", "sycamore", "tests"))


def test_get_empty_intervention():
    empty_dict = get_all_interventions_dict("empty_intervention_data.json")

    assert empty_dict == {}

def test_get_intervention():
    interventions_dict = get_all_interventions_dict(
        "sample_intervention_data.json")

    assert type(interventions_dict["ntz4apjf"]) is dict

    assert interventions_dict["ntz4apjf"]["Time_0"] == "2021-12-14"


def test_sample_agg_data():
    sample_agg_data = aggregate_surveys("sample_dir", ["idr8gqdh"])

    assert pd.isnull(sample_agg_data.loc[0, "time_prev"])

    assert all(pd.to_datetime(sample_agg_data['timestamp'], unit='ms') ==
               sample_agg_data['UTC time'])


def test_gen_survey_schedule():
    interventions_dict = get_all_interventions_dict(
        "sample_intervention_data.json")

    sample_schedule = gen_survey_schedule(
        "sample_study_surveys_and_settings.json",
        time_start=pd.to_datetime("2021-12-01"),
        time_end=pd.to_datetime("2021-12-30"),
        beiwe_ids=["idr8gqdh"],
        all_interventions_dict=interventions_dict)

    assert sample_schedule.shape[0] == 7

    assert np.mean(
        sample_schedule.columns ==
        pd.Index(['delivery_time','next_delivery_time', 'id', 'beiwe_id',
                  'question_id'])
    ) == 1.0
