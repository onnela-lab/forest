import numpy as np
import os
import pandas as pd


from forest.sycamore.survey_config import (get_all_interventions_dict,
                                           survey_submits_no_config)
from forest.sycamore.functions import (aggregate_surveys,
                                       aggregate_surveys_no_config,
                                       aggregate_surveys_config)
from forest.sycamore.survey_config import gen_survey_schedule, survey_submits
from forest.sycamore.changed_answers import agg_changed_answers_summary

TEST_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def test_get_empty_intervention():
    filepath = os.path.join(TEST_DATA_DIR, "empty_intervention_data.json")
    empty_dict = get_all_interventions_dict(filepath)

    assert empty_dict == {}


def test_get_intervention():
    filepath = os.path.join(TEST_DATA_DIR, "sample_intervention_data.json")
    interventions_dict = get_all_interventions_dict(filepath)

    assert type(interventions_dict["ntz4apjf"]) is dict

    assert interventions_dict["ntz4apjf"]["Time_0"] == "2021-12-14"


def test_aggregate_surveys():
    filepath = os.path.join(TEST_DATA_DIR, "sample_dir")
    sample_agg_data = aggregate_surveys(filepath, ["idr8gqdh"])

    assert pd.isnull(sample_agg_data.loc[0, "time_prev"])

    assert np.mean(
        pd.to_datetime(sample_agg_data["timestamp"], unit="ms"
                       ) == sample_agg_data["UTC time"]
    ) == 1.0
    assert "MALFORMED" not in sample_agg_data["question text"].values


def test_gen_survey_schedule():
    interventions_path = os.path.join(TEST_DATA_DIR,
                                      "sample_intervention_data.json")
    interventions_dict = get_all_interventions_dict(interventions_path)
    surveys_settings_path = os.path.join(
        TEST_DATA_DIR, "sample_study_surveys_and_settings.json"
    )

    sample_schedule = gen_survey_schedule(
        surveys_settings_path,
        time_start=pd.to_datetime("2021-12-01"),
        time_end=pd.to_datetime("2021-12-30"),
        beiwe_ids=["idr8gqdh"],
        all_interventions_dict=interventions_dict)

    assert sample_schedule.shape[0] == 7

    assert np.mean(
        sample_schedule.columns ==
        pd.Index(["delivery_time", "next_delivery_time", "id", "beiwe_id",
                  "question_id"])
    ) == 1.0


def test_aggregate_surveys_no_config():
    filepath = os.path.join(TEST_DATA_DIR, "sample_dir")
    agg_data = aggregate_surveys_no_config(filepath, "UTC")
    assert np.mean(pd.to_datetime(agg_data["timestamp"], unit="ms") ==
                   agg_data["UTC time"]) == 1.0
    assert len(agg_data.surv_inst_flg.unique()) == 2
    assert len(agg_data.DOW.unique()) == 2


def test_aggregate_surveys_config():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    survey_settings_path = os.path.join(
        TEST_DATA_DIR, "sample_study_surveys_and_settings.json"
    )

    agg_data = aggregate_surveys_config(study_dir, survey_settings_path, "UTC")
    assert np.mean(pd.to_datetime(agg_data["timestamp"], unit="ms") ==
                   agg_data["UTC time"]) == 1.0
    assert len(agg_data.surv_inst_flg.unique()) == 2
    assert len(agg_data.DOW.unique()) == 2


def test_agg_changed_answers_summary():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    survey_settings_path = os.path.join(
        TEST_DATA_DIR, "sample_study_surveys_and_settings.json"
    )
    agg_data = aggregate_surveys_config(study_dir, survey_settings_path, "UTC")
    ca_detail, ca_summary = agg_changed_answers_summary(
        survey_settings_path, agg_data
    )
    assert ca_detail.shape[0] == 5
    assert ca_detail.shape[1] == 14
    assert ca_summary.shape[0] == 4
    assert ca_summary.shape[1] == 9


def test_survey_submits_with_no_submissions():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    survey_settings_path = os.path.join(
        TEST_DATA_DIR, "sample_study_surveys_and_settings.json"
    )
    interventions_path = os.path.join(
        TEST_DATA_DIR, "sample_intervention_data.json"
    )
    agg_data = aggregate_surveys_config(study_dir, survey_settings_path, "UTC")
    interventions_dict = get_all_interventions_dict(interventions_path)
    ss_detail, ss_summary = survey_submits(
        survey_settings_path, "2021-12-01", "2021-12-30",
        ["idr8gqdh"], agg_data, interventions_dict
    )
    assert ss_detail.shape[0] == 0
    assert ss_summary.shape[0] == 0


def test_survey_submits_no_config():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    submits_tbl = survey_submits_no_config(study_dir, "UTC")
    assert submits_tbl.shape[0] == 2
