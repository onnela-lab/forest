import os

import numpy as np
import pandas as pd
import pytest

from forest.constants import Frequency
from forest.sycamore.common import (aggregate_surveys,
                                    aggregate_surveys_no_config,
                                    aggregate_surveys_config,
                                    read_user_answers_stream,
                                    read_aggregate_answers_stream,
                                    get_choices_with_sep_values)
from forest.sycamore.utils import filename_to_timestamp
from forest.sycamore.submits import (gen_survey_schedule,
                                     get_all_interventions_dict,
                                     survey_submits, summarize_submits,
                                     survey_submits_no_config)
from forest.sycamore.responses import (agg_changed_answers_summary,
                                       format_responses_by_submission)
from forest.sycamore.read_audio import (get_audio_survey_id_dict,
                                        get_config_id_dict,
                                        read_user_audio_recordings_stream,
                                        read_aggregate_audio_recordings_stream)


TEST_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

INTERVENTIONS_PATH = os.path.join(TEST_DATA_DIR,
                                  "sample_intervention_data.json")

SAMPLE_DIR = os.path.join(TEST_DATA_DIR, "sample_dir")

SURVEY_SETTINGS_PATH = os.path.join(TEST_DATA_DIR,
                                    "sample_study_surveys_and_settings.json")

SURVEY_SETTINGS_PATH_FOR_SUBMITS = os.path.join(TEST_DATA_DIR,
                                                "config_file_for_submits.json")

CONFIG_WITH_SEPS = os.path.join(TEST_DATA_DIR,
                                "config_file_with_commas_and_semicolons.json")

HISTORY_WITH_SEPS = os.path.join(
    TEST_DATA_DIR, "history_file_with_commas_and_semicolons.json"
)

AUDIO_SURVEY_CONFIG = os.path.join(
    TEST_DATA_DIR, "audio_survey_config.json"
)

AUDIO_SURVEY_HISTORY = os.path.join(
    TEST_DATA_DIR, "audio_survey_history.json"
)

SEP_QS_DIR = os.path.join(TEST_DATA_DIR, "dir_with_seps_in_qs")


@pytest.fixture
def agg_data_config():
    return aggregate_surveys_config(SAMPLE_DIR, SURVEY_SETTINGS_PATH,
                                    "UTC", users=["16au2moz", "idr8gqdh"])


@pytest.fixture
def agg_data_no_config():
    return aggregate_surveys_no_config(SAMPLE_DIR, study_tz="UTC",
                                       users = ["16au2moz", "idr8gqdh"])


@pytest.fixture
def submits_data():
    agg_data = aggregate_surveys_config(
        SAMPLE_DIR, SURVEY_SETTINGS_PATH_FOR_SUBMITS, study_tz="UTC",
        users=["16au2moz", "idr8gqdh"]
    )
    return survey_submits(SURVEY_SETTINGS_PATH_FOR_SUBMITS,
                          "2021-12-01", "2022-04-30", ["idr8gqdh", "16au2moz"],
                          agg_data, INTERVENTIONS_PATH)


def test_survey_submits(submits_data):
    assert submits_data.shape[0] == 334


def test_summarize_submits(submits_data):
    summary = summarize_submits(submits_data)
    assert summary.shape[0] == 2


def test_summarize_submits_hour(submits_data):
    summary = summarize_submits(submits_data, Frequency.HOURLY)
    assert summary.shape[0] == 292


def test_summarize_submits_day(submits_data):
    summary = summarize_submits(submits_data, timeunit=Frequency.DAILY)
    assert summary.shape[0] == 208


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
        users=["idr8gqdh"],
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
        users=["idr8gqdh"],
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
        users=["idr8gqdh"],
        all_interventions_dict=interventions_dict)
    # should include one survey schedule and cutoff the other one
    assert sample_schedule.shape[0] == 1
    assert np.mean(
        sample_schedule.columns ==
        pd.Index(["delivery_time", "next_delivery_time", "id", "beiwe_id",
                  "question_id"])
    ) == 1.0


def test_aggregate_surveys_no_config():
    agg_data = aggregate_surveys_no_config(SAMPLE_DIR, study_tz="UTC",
                                           users=["16au2moz", "idr8gqdh"])
    assert agg_data.shape[0] == 50
    assert len(agg_data.DOW.unique()) == 4


def test_aggregate_surveys_config():
    agg_data = aggregate_surveys_config(SAMPLE_DIR, SURVEY_SETTINGS_PATH,
                                        "UTC", users = ["16au2moz", "idr8gqdh"])
    assert agg_data.shape[0] == 50
    assert len(agg_data.DOW.unique()) == 4


def test_agg_changed_answers_summary(agg_data_config):
    ca_detail, ca_summary = agg_changed_answers_summary(
        SURVEY_SETTINGS_PATH, agg_data_config
    )
    assert ca_detail.shape[0] == 12
    assert ca_detail.shape[1] == 14
    assert ca_summary.shape[0] == 7
    assert ca_summary.shape[1] == 9


def test_agg_changed_answers_summary_no_config(agg_data_no_config):
    # make sure that changed_answers_summary is the same if we aggregate the
    # data using the no config function
    ca_detail, ca_summary = agg_changed_answers_summary(
        SURVEY_SETTINGS_PATH, agg_data_no_config
    )
    assert ca_detail.shape[0] == 12
    assert ca_detail.shape[1] == 14
    assert ca_summary.shape[0] == 7
    assert ca_summary.shape[1] == 9


def test_survey_submits_with_no_submissions(agg_data_config):
    ss_detail = survey_submits(
        SURVEY_SETTINGS_PATH, "2021-12-01", "2021-12-30",
        ["idr8gqdh"], agg_data_config, INTERVENTIONS_PATH
    )
    assert ss_detail.shape[0] == 0


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
    assert df.shape[1] == 14


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
    agg_data = aggregate_surveys_config(empty_dir, SURVEY_SETTINGS_PATH, "UTC",
                                        users=["16au2moz", "idr8gqdh"])
    assert agg_data.shape[0] == 0


def test_aggregate_surveys_no_config_empty_dir():
    empty_dir = os.path.join(TEST_DATA_DIR, "empty_dir")
    agg_data_no_config = aggregate_surveys_no_config(
        empty_dir, "UTC", users=["16au2moz", "idr8gqdh"]
    )
    assert agg_data_no_config.shape[0] == 0


def test_aggregate_surveys_config_time_no_files():
    agg_data = aggregate_surveys_config(
        SAMPLE_DIR, SURVEY_SETTINGS_PATH, "UTC", time_start="2008-01-01",
        time_end="2008-05-01", users=["16au2moz", "idr8gqdh"]
    )
    assert agg_data.shape[0] == 0


def test_aggregate_surveys_no_config_time_no_files():
    agg_data_no_config = aggregate_surveys_no_config(
        SAMPLE_DIR, "UTC", time_start="2008-01-01", time_end="2008-05-01",
        users=["16au2moz", "idr8gqdh"]
    )
    assert agg_data_no_config.shape[0] == 0


def test_aggregate_surveys_config_restriction_start():
    agg_data = aggregate_surveys_config(
        SAMPLE_DIR, SURVEY_SETTINGS_PATH, "UTC", time_start="2022-03-12",
        time_end="2022-04-01", users=["16au2moz", "idr8gqdh"]
    )
    assert agg_data.shape[0] == 12
    assert np.mean(agg_data["Local time"] > pd.to_datetime("2022-03-12")) == 1


def test_aggregate_surveys_no_config_restriction_start():
    agg_data_no_config = aggregate_surveys_no_config(
        SAMPLE_DIR, "UTC", time_start="2022-03-12", time_end="2022-04-01",
        users=["16au2moz", "idr8gqdh"]
    )
    assert agg_data_no_config.shape[0] == 12
    assert np.mean(
        agg_data_no_config["Local time"] > pd.to_datetime("2022-03-12")
    ) == 1


def test_aggregate_surveys_config_restriction_end():
    agg_data = aggregate_surveys_config(
        SAMPLE_DIR, SURVEY_SETTINGS_PATH, "UTC", time_start="2001-01-01",
        time_end="2022-03-12", users=["16au2moz", "idr8gqdh"]
    )
    assert agg_data.shape[0] == 38
    assert np.mean(agg_data["Local time"] < pd.to_datetime("2022-03-12")) == 1


def test_aggregate_surveys_no_config_restriction_end():
    agg_data_no_config = aggregate_surveys_no_config(
        SAMPLE_DIR, "UTC", time_start="2001-01-01", time_end="2022-03-12",
        users=["16au2moz", "idr8gqdh"]
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


def test_get_choices_with_sep_values_config():
    qs_with_seps = get_choices_with_sep_values(CONFIG_WITH_SEPS, None)
    assert len(qs_with_seps.keys()) == 1
    assert len(qs_with_seps['07369e05-b2f7-465a-e66e-8e473fcd3c2f']) == 5


def test_get_choices_with_sep_values_history():
    qs_with_seps = get_choices_with_sep_values(None, HISTORY_WITH_SEPS)
    assert len(qs_with_seps.keys()) == 2
    assert len(qs_with_seps['07369e05-b2f7-465a-e66e-8e473fcd3c2f']) == 5
    assert len(qs_with_seps['90268105-b59e-4e17-d231-613e8523e310']) == 2


def test_get_choices_with_sep_values_both():
    qs_with_seps = get_choices_with_sep_values(CONFIG_WITH_SEPS,
                                               HISTORY_WITH_SEPS)
    assert len(qs_with_seps.keys()) == 2
    assert len(qs_with_seps['07369e05-b2f7-465a-e66e-8e473fcd3c2f']) == 5
    assert len(qs_with_seps['90268105-b59e-4e17-d231-613e8523e310']) == 2


def test_aggregate_surveys_config_using_sep_data():
    agg_data = aggregate_surveys_config(
        SEP_QS_DIR, CONFIG_WITH_SEPS,"UTC", history_path=HISTORY_WITH_SEPS)
    assert agg_data.shape[0] == 19  # 4 lines (3 answers and a submit line) for
    # each survey in survey_answers, plus 3 from the survey_timings file after
    # the delivery line is removed
    assert agg_data.loc[3, "answer"] == "here (, ) is a comma"
    assert agg_data.loc[6, "answer"] == ", comma at begin"
    assert agg_data.loc[9, "answer"] == "comma at end , "
    assert agg_data.loc[14, "answer"] == "no problems here"

def test_get_audio_survey_id_dict():
    audio_survey_id_dict = get_audio_survey_id_dict(AUDIO_SURVEY_HISTORY)
    assert set(audio_survey_id_dict.keys()) == {'prompt1', 'prompt2', 'prompt3'}
    assert audio_survey_id_dict['prompt1'] == "tO1GFjGJjMnaDRThUQK6l4dv"
    assert audio_survey_id_dict['prompt2'] == "6iWVNrsd1RE2zAeIPegZDrCc"
    assert audio_survey_id_dict['prompt3'] =="oB7h0i0GwCK2sviY1DRXzHIe"

def test_get_config_id_dict():
    config_id_dict = get_config_id_dict(AUDIO_SURVEY_CONFIG)
    assert set(config_id_dict.keys()) == {'prompt1', 'prompt2', 'prompt3'}
    assert config_id_dict['prompt1'] == 1
    assert config_id_dict['prompt2'] == 3
    assert config_id_dict['prompt3'] == 6

def test_read_user_audio_recordings_stream():
    df = read_user_audio_recordings_stream(
        SAMPLE_DIR, "audioqdz", history_path = AUDIO_SURVEY_HISTORY
    )
    assert df.shape[0] == 16
    assert df["UTC time"].nunique() == 8
    assert df["survey id"].nunique() == 2
    assert df["question text"].nunique() == 2

def test_read_user_audio_recordings_stream_no_history():
    df = read_user_audio_recordings_stream(
        SAMPLE_DIR, "audioqdz"
    )
    assert df.shape[0] == 16
    assert df["UTC time"].nunique() == 8
    assert df["question text"].nunique() == 1
    assert df["survey id"].nunique() == 2


def test_read_aggregate_audio_recordings_stream():
    df = read_aggregate_audio_recordings_stream(
        SAMPLE_DIR, history_path=AUDIO_SURVEY_HISTORY
    )
    assert df.shape[0] == 26
    assert df["UTC time"].nunique() == 8
    assert df["survey id"].nunique() == 2
    assert df["question text"].nunique() == 2
    assert df["beiwe_id"].nunique() == 2


def test_read_aggregate_audio_recordings_stream_no_history():
    df = read_aggregate_audio_recordings_stream(SAMPLE_DIR)
    assert df.shape[0] == 26
    assert df["UTC time"].nunique() == 8
    assert df["survey id"].nunique() == 2
    assert df["question text"].nunique() == 1 #should only have "UNKNOWN"
    assert df["question text"].unique().tolist() == ["UNKNOWN"]
    assert df["beiwe_id"].nunique() == 2
