import os

import numpy as np
import pandas as pd
import pytest

from forest.sycamore.submits import (get_all_interventions_dict,
                                     survey_submits_no_config)
from forest.sycamore.common import (aggregate_surveys,
                                    aggregate_surveys_no_config,
                                    aggregate_surveys_config,
                                    filename_to_timestamp,
                                    get_subdirs,
                                    read_user_answers_stream,
                                    read_aggregate_answers_stream)
from forest.sycamore.submits import gen_survey_schedule, survey_submits
from forest.sycamore.responses import (agg_changed_answers_summary,
                                       format_responses_by_submission)


TEST_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

INTERVENTIONS_PATH = os.path.join(TEST_DATA_DIR,
                                  "sample_intervention_data.json")

SAMPLE_DIR = os.path.join(TEST_DATA_DIR, "sample_dir")

SURVEY_SETTINGS_PATH = os.path.join(TEST_DATA_DIR,
                                    "sample_study_surveys_and_settings.json")


@pytest.fixture
def agg_data_config():
    return aggregate_surveys_config(SAMPLE_DIR, SURVEY_SETTINGS_PATH,
                                    "UTC")


@pytest.fixture
def agg_data_no_config():
    return aggregate_surveys_no_config(SAMPLE_DIR, study_tz="UTC")


def test_get_empty_intervention():
    empty_path = os.path.join(TEST_DATA_DIR, "empty_intervention_data.json")
    empty_dict = get_all_interventions_dict(empty_path)

    assert empty_dict == {}


def test_get_intervention():
    interventions_dict = get_all_interventions_dict(INTERVENTIONS_PATH)
    assert type(interventions_dict["ntz4apjf"]) is dict
    assert interventions_dict["ntz4apjf"]["Time_0"] == "2021-12-14"


def test_aggregate_surveys():
    sample_agg_data = aggregate_surveys(SAMPLE_DIR, ["idr8gqdh"])

    assert pd.isnull(sample_agg_data.loc[0, "time_prev"])
    assert "MALFORMED" not in sample_agg_data["question text"].values


def test_gen_survey_schedule():
    interventions_dict = get_all_interventions_dict(INTERVENTIONS_PATH)
    sample_schedule = gen_survey_schedule(
        SURVEY_SETTINGS_PATH,
        time_start=pd.to_datetime("2021-12-01"),
        time_end=pd.to_datetime("2021-12-30"),
        beiwe_ids=["idr8gqdh"],
        all_interventions_dict=interventions_dict)

    assert sample_schedule.shape[0] == 6
    assert np.mean(
        sample_schedule.columns ==
        pd.Index(["delivery_time", "next_delivery_time", "id", "beiwe_id",
                  "question_id"])
    ) == 1.0


def test_gen_survey_schedule_one_weekly():
    interventions_dict = get_all_interventions_dict(INTERVENTIONS_PATH)
    sample_schedule = gen_survey_schedule(
        os.path.join(TEST_DATA_DIR, "one_weekly_survey.json"),
        time_start=pd.to_datetime("2021-12-01"),
        time_end=pd.to_datetime("2021-12-08"),
        beiwe_ids=["idr8gqdh"],
        all_interventions_dict=interventions_dict)
    assert sample_schedule.shape[0] == 1
    assert np.mean(
        sample_schedule.columns ==
        pd.Index(["delivery_time", "next_delivery_time", "id", "beiwe_id",
                  "question_id"])
    ) == 1.0


def test_gen_survey_schedule_cutoff_relative():
    interventions_dict = get_all_interventions_dict(INTERVENTIONS_PATH)
    sample_schedule = gen_survey_schedule(
        os.path.join(TEST_DATA_DIR, "far_relative_survey.json"),
        time_start=pd.to_datetime("2021-12-01"),
        time_end=pd.to_datetime("2021-12-08"),
        beiwe_ids=["idr8gqdh"],
        all_interventions_dict=interventions_dict)
    # should include one survey schedule and cutoff the other one
    assert sample_schedule.shape[0] == 1
    assert np.mean(
        sample_schedule.columns ==
        pd.Index(["delivery_time", "next_delivery_time", "id", "beiwe_id",
                  "question_id"])
    ) == 1.0


def test_aggregate_surveys_no_config():
    agg_data = aggregate_surveys_no_config(SAMPLE_DIR, study_tz="UTC")
    assert agg_data.shape[0] == 50
    assert len(agg_data.DOW.unique()) == 4


def test_aggregate_surveys_config():
    agg_data = aggregate_surveys_config(SAMPLE_DIR, SURVEY_SETTINGS_PATH,
                                        "UTC")
    assert agg_data.shape[0] == 50
    assert len(agg_data.DOW.unique()) == 4


def test_agg_changed_answers_summary(agg_data_config):
    ca_detail, ca_summary = agg_changed_answers_summary(
        SURVEY_SETTINGS_PATH, agg_data_config
    )
    assert ca_detail.shape[0] == 8
    assert ca_detail.shape[1] == 14
    assert ca_summary.shape[0] == 7
    assert ca_summary.shape[1] == 9


def test_agg_changed_answers_summary_no_config(agg_data_no_config):
    # make sure that changed_answers_summary is the same if we aggregate the
    # data using the no config function
    ca_detail, ca_summary = agg_changed_answers_summary(
        SURVEY_SETTINGS_PATH, agg_data_no_config
    )
    assert ca_detail.shape[0] == 8
    assert ca_detail.shape[1] == 14
    assert ca_summary.shape[0] == 7
    assert ca_summary.shape[1] == 9


def test_survey_submits_with_no_submissions(agg_data_config):
    ss_detail, ss_summary = survey_submits(
        SURVEY_SETTINGS_PATH, "2021-12-01", "2021-12-30",
        ["idr8gqdh"], agg_data_config, INTERVENTIONS_PATH
    )
    assert ss_detail.shape[0] == 0
    assert ss_summary.shape[0] == 0


def test_survey_submits_no_config_adc(agg_data_config):
    # Ensure that survey_submits_no_config generates the same information
    # regardless of whether the passed data was generated from a config file.
    submits_tbl = survey_submits_no_config(agg_data_config)
    assert submits_tbl.shape[0] == 6


def test_survey_submits_no_config_adnc(agg_data_no_config):
    submits_tbl = survey_submits_no_config(agg_data_no_config)
    assert submits_tbl.shape[0] == 6


def test_read_user_answers_stream():
    df = read_user_answers_stream(SAMPLE_DIR, "idr8gqdh", "UTC")
    assert len(df["Local time"].unique()) == 2


def test_read_aggregate_answers_stream():
    df = read_aggregate_answers_stream(SAMPLE_DIR)
    assert len(df["beiwe_id"].unique()) == 2
    assert df.shape[1] == 11


def test_format_responses_by_submission_adnc(agg_data_no_config):
    surveys_dict = format_responses_by_submission(agg_data_no_config)
    assert len(surveys_dict.keys()) == 1
    assert surveys_dict["hkmxse2N7aMGfNyVMQDiWWEP"].shape[0] == 10


def test_format_responses_by_submission_adc(agg_data_config):
    surveys_dict = format_responses_by_submission(agg_data_config)
    assert len(surveys_dict.keys()) == 1
    assert surveys_dict["hkmxse2N7aMGfNyVMQDiWWEP"].shape[0] == 10


def test_aggregate_surveys_config_empty_dir():
    empty_dir = os.path.join(TEST_DATA_DIR, "empty_dir")
    agg_data = aggregate_surveys_config(empty_dir, SURVEY_SETTINGS_PATH, "UTC")
    assert agg_data.shape[0] == 0


def test_aggregate_surveys_no_config_empty_dir():
    empty_dir = os.path.join(TEST_DATA_DIR, "empty_dir")
    agg_data_no_config = aggregate_surveys_no_config(empty_dir, "UTC")
    assert agg_data_no_config.shape[0] == 0


def test_aggregate_surveys_config_time_no_files():
    agg_data = aggregate_surveys_config(
        SAMPLE_DIR, SURVEY_SETTINGS_PATH, "UTC", time_start="2008-01-01",
        time_end="2008-05-01"
    )
    assert agg_data.shape[0] == 0


def test_aggregate_surveys_no_config_time_no_files():
    agg_data_no_config = aggregate_surveys_no_config(
        SAMPLE_DIR, "UTC", time_start="2008-01-01", time_end="2008-05-01"
    )
    assert agg_data_no_config.shape[0] == 0


def test_aggregate_surveys_config_restriction_start():
    agg_data = aggregate_surveys_config(
        SAMPLE_DIR, SURVEY_SETTINGS_PATH, "UTC", time_start="2022-03-12",
        time_end="2022-04-01"
    )
    assert agg_data.shape[0] == 12
    assert np.mean(agg_data["Local time"] > pd.to_datetime("2022-03-12")) == 1


def test_aggregate_surveys_no_config_restriction_start():
    agg_data_no_config = aggregate_surveys_no_config(
        SAMPLE_DIR, "UTC", time_start="2022-03-12", time_end="2022-04-01"
    )
    assert agg_data_no_config.shape[0] == 12
    assert np.mean(
        agg_data_no_config["Local time"] > pd.to_datetime("2022-03-12")
    ) == 1


def test_aggregate_surveys_config_restriction_end():
    agg_data = aggregate_surveys_config(
        SAMPLE_DIR, SURVEY_SETTINGS_PATH, "UTC", time_start="2001-01-01",
        time_end="2022-03-12"
    )
    assert agg_data.shape[0] == 38
    assert np.mean(agg_data["Local time"] < pd.to_datetime("2022-03-12")) == 1


def test_aggregate_surveys_no_config_restriction_end():
    agg_data_no_config = aggregate_surveys_no_config(
        SAMPLE_DIR, "UTC", time_start="2001-01-01", time_end="2022-03-12"
    )
    assert agg_data_no_config.shape[0] == 38
    assert np.mean(
        agg_data_no_config["Local time"] < pd.to_datetime("2022-03-12")
    ) == 1


def test_file_to_datetime():
    test_str = "2022-03-14 16_32_56+00_00.csv"
    expected_timestamp = pd.to_datetime("2022-03-14 16:32:56")
    assert filename_to_timestamp(test_str, "UTC") == expected_timestamp
    test_str2 = "2022-03-14 16_32_56+00_00.csv"
    assert filename_to_timestamp(test_str2, "UTC") == expected_timestamp
    test_str3 = "2022-03-14 17_32_56+01_00.csv"
    assert filename_to_timestamp(test_str3, "UTC") == expected_timestamp
    test_str4 = "2022-03-14 17_32_56+01_00_1.csv"
    assert filename_to_timestamp(test_str4, "UTC") == expected_timestamp


def test_get_subdirs():
    users_list = get_subdirs(SAMPLE_DIR)
    assert len(users_list) == 2
