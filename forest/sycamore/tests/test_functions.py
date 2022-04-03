import numpy as np
import os
import pandas as pd


from forest.sycamore.submits import (get_all_interventions_dict,
                                     survey_submits_no_config)
from forest.sycamore.common import (aggregate_surveys,
                                    aggregate_surveys_no_config,
                                    aggregate_surveys_config,
                                    read_one_answers_stream,
                                    read_aggregate_answers_stream)
from forest.sycamore.submits import gen_survey_schedule, survey_submits
from forest.sycamore.responses import (agg_changed_answers_summary,
                                       by_survey_administration)


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
    agg_data = aggregate_surveys_no_config(filepath, study_tz="UTC")
    assert agg_data.shape[0] == 52
    assert len(agg_data.DOW.unique()) == 4


def test_aggregate_surveys_config():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    survey_settings_path = os.path.join(
        TEST_DATA_DIR, "sample_study_surveys_and_settings.json"
    )

    agg_data = aggregate_surveys_config(study_dir, survey_settings_path, "UTC")
    assert agg_data.shape[0] == 52
    assert len(agg_data.DOW.unique()) == 4


def test_agg_changed_answers_summary():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    survey_settings_path = os.path.join(
        TEST_DATA_DIR, "sample_study_surveys_and_settings.json"
    )
    agg_data = aggregate_surveys_config(study_dir, survey_settings_path, "UTC")
    ca_detail, ca_summary = agg_changed_answers_summary(
        survey_settings_path, agg_data
    )
    assert ca_detail.shape[0] == 8
    assert ca_detail.shape[1] == 14
    assert ca_summary.shape[0] == 7
    assert ca_summary.shape[1] == 9

def test_agg_changed_answers_summary_no_config():
    # make sure that changed_answers_summary is the same if we aggregate the
    # data using the no config function
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    survey_settings_path = os.path.join(
        TEST_DATA_DIR, "sample_study_surveys_and_settings.json"
    )
    agg_data = aggregate_surveys_no_config(study_dir, "UTC")
    ca_detail, ca_summary = agg_changed_answers_summary(
        survey_settings_path, agg_data
    )
    assert ca_detail.shape[0] == 8
    assert ca_detail.shape[1] == 14
    assert ca_summary.shape[0] == 7
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
    # Ensure that survey_submits_no_config generates the same information
    # regardless of whether the passed data was generated from a config file.
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    survey_settings_path = os.path.join(
        TEST_DATA_DIR, "sample_study_surveys_and_settings.json"
    )
    agg_data = aggregate_surveys_config(study_dir, survey_settings_path, "UTC")
    submits_tbl = survey_submits_no_config(agg_data)
    assert submits_tbl.shape[0] == 6

def test_survey_submits_no_config_with_config_input():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    agg_data = aggregate_surveys_no_config(study_dir, study_tz="UTC")
    submits_tbl = survey_submits_no_config(agg_data)
    assert submits_tbl.shape[0] == 6


def test_read_one_answers_stream():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    df = read_one_answers_stream(study_dir, "idr8gqdh", "UTC")
    assert len(df["Local time"].unique()) == 2


def test_read_aggregate_answers_stream():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    df = read_aggregate_answers_stream(study_dir)
    assert len(df["beiwe_id"].unique()) == 2
    assert df.shape[1] == 10

def test_by_survey_administration():
    study_dir = os.path.join(TEST_DATA_DIR, "sample_dir")
    agg_data = aggregate_surveys_no_config(study_dir, study_tz="UTC")
    surveys_dict = by_survey_administration(agg_data)
    assert len(surveys_dict.keys()) == 1
    assert surveys_dict["hkmxse2N7aMGfNyVMQDiWWEP"].shape[0] == 10

