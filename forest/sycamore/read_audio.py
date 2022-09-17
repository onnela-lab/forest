"""Functions associated with reading audio surveys"""

import logging
import os

import pandas as pd
import numpy as np

from forest.utils import get_ids
from forest.sycamore.constants import EARLIEST_DATE
from forest.sycamore.utils import (read_json, get_month_from_today,
                                   filename_to_timestamp)


logger = logging.getLogger(__name__)


def get_audio_survey_id_dict(history_file: str) -> dict:
    """A dict that goes from the most recent prompt to the survey ID for audio
        surveys
    Args:
        history_file:
    Returns:
        dictionary with keys for each prompt, and values with survey IDs"""
    output_dict = dict()
    if history_file is None:
        return output_dict
    history_dict = read_json(history_file)
    for key in history_dict.keys():
        most_recent_update = history_dict[key][-1]['survey_json']
        if most_recent_update is None:
            continue
        if type(most_recent_update) != list:
            continue
        if len(most_recent_update) == 0:
            continue
        if "prompt" not in most_recent_update[0].keys():
            continue
        output_dict[most_recent_update[0]['prompt']] = key

    return output_dict


def get_config_id_dict(config_path: str) -> dict:
    """Get a dict with question prompts as keys and the config IDs as values
    Args:
        config_path: Path to survey config JSON file
    Returns:
        dict with a key for each question prompt, and the config ID (the order
        of the question in the config file) as values
    """
    output_dict = dict()
    if config_path is None:
        return output_dict
    surveys = read_json(config_path)["surveys"]
    for i, s in enumerate(surveys):
        if "content" not in s.keys():
            continue
        if type(s["content"]) is not list:
            continue
        if len(s["content"]) < 1:
            continue
        if "prompt" not in s["content"][0]:
            continue

        output_dict[s["content"][0]["prompt"]] = i
    return output_dict


def read_user_audio_recordings_stream(
        download_folder: str, user: str, tz_str: str = "UTC",
        time_start: str = EARLIEST_DATE, time_end: str = None,
        history_path: str = None
) -> pd.DataFrame:
    """Reads in all audio_recordings data for a user

    Reads survey_answers data and creates a column with the survey
    ID, as well as a column for the date from the filename.

    Args:
        download_folder:
            path to downloaded data. A folder wiith the user ID should be a
            subdirectory of this path.
        user:
            ID of user to aggregate data
        tz_str:
            Time Zone to include in Local time column of output. See
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for
            options
        time_start:
            The first date of the survey data, in YYYY-MM-DD format
        time_end:
            The last date of the survey data, in YYYY-MM-DD format
        history_path: Filepath to the survey history file. If this is not
            included, audio survey prompts will not be included.

    Returns:
        DataFrame with stacked data, a field for the beiwe ID, a field for the
        survey, and a filed with the time in the filename.
    """
    audio_survey_id_dict = get_audio_survey_id_dict(history_path)
    if time_end is None:
        time_end = get_month_from_today()
    audio_dir = os.path.join(download_folder, user, "audio_recordings")
    if os.path.isdir(audio_dir):
        # get all survey IDs included for this user (data will have one folder
        # per survey)
        survey_ids = get_ids(audio_dir)
        all_surveys = []
        timestamp_start = pd.to_datetime(time_start)
        timestamp_end = pd.to_datetime(time_end)
        for survey in survey_ids:
            # get all audio files in the survey subdirectory
            all_files = []
            for filepath in os.listdir(os.path.join(audio_dir, survey)):
                filename = os.path.basename(filepath)
                valid_file = (filepath.endswith(".wav")
                              or filepath.endswith(".mp4")
                              and (timestamp_start
                                   < filename_to_timestamp(filename, tz_str)
                                   < timestamp_end))
                if valid_file:
                    all_files.append(filepath)

            if len(all_files) == 0:
                logger.warning("No audio_recordings for user %s in given time "
                               "frames.", user)
                return pd.DataFrame(columns=["Local time"],
                                    dtype="datetime64[ns]")

            survey_dfs = []
            # We want to be able to process the surveys even if they didn't
            # include the survey history file, but if they did include the
            # survey history file, we want to have the prompt for readability
            survey_prompt = "UNKNOWN"
            for prompt in audio_survey_id_dict.keys():
                if audio_survey_id_dict[prompt] == survey:
                    survey_prompt = prompt

            # We need to enumerate to tell different survey occasions apart
            for i, file in enumerate(all_files):
                filename = os.path.basename(file)
                current_df = pd.DataFrame({
                    "UTC time": [filename_to_timestamp(filename, "UTC")] * 2,
                    "survey id": [survey] * 2,
                    "question_id": [survey] * 2,
                    "answer": ["audio recording", ""],
                    "question type": ["audio recording", ""],
                    "question text": [survey_prompt] * 2,
                    "question answer options": ["audio recording", ""],
                    "submit_line": [0, 1],  # one of the lines will be a submit
                    # line
                    "surv_inst_flg": [i] * 2
                })
                survey_dfs.append(current_df)
            if len(survey_dfs) == 0:
                logger.warning("No survey_answers for user %s.", user)
                return pd.DataFrame(columns=["Local time"],
                                    dtype="datetime64[ns]")
            survey_data = pd.concat(survey_dfs, axis=0, ignore_index=True)
            survey_data["beiwe_id"] = user
            survey_data["Local time"] = survey_data[
                "UTC time"
            ].dt.tz_localize("UTC").dt.tz_convert(tz_str).dt.tz_localize(None)

            # Add question index column to make things look like the survey
            # timings stream. We do not need to worry about verifying that the
            # question IDs on the new lines are different as we did for survey
            # timings because these are all final submissions.
            survey_data["question index"] = 1
            survey_data["question index"] = survey_data.groupby(
                ["survey id", "beiwe_id"]
            )["question index"].cumsum()

            all_surveys.append(survey_data)
        if len(all_surveys) == 0:
            return pd.DataFrame(columns=["Local time"], dtype="datetime64[ns]")
        return pd.concat(all_surveys, axis=0, ignore_index=True)
    else:
        logger.warning("No survey_answers for user %s.", user)
        return pd.DataFrame(columns=["Local time"], dtype="datetime64[ns]")


def read_aggregate_audio_recordings_stream(
        download_folder: str, users: list = None,
        tz_str: str = "UTC", config_path: str = None,
        time_start: str = EARLIEST_DATE, time_end: str = None,
        history_path: str = None
) -> pd.DataFrame:
    """Reads in all answers data for many users and fixes Android users to have
    an answer instead of an integer

    Args:
        download_folder:
            path to downloaded data. This folder should have Beiwe IDs as
            subdirectories.
        users:
            List of IDs of users to aggregate data on
        tz_str:
            Time Zone to include in Local time column of output. See
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for
            options
        config_path:
            Path to config file. If this is included, the function
            uses the config file to resolve semicolons that appear in survey
            answers lists. If this is not included, the function attempt to use
            iPhone responses to resolve semicolons.
        time_start:
            The first date of the survey data, in YYYY-MM-DD format
        time_end:
            The last date of the survey data, in YYYY-MM-DD format
        history_path:
            Path to survey history file. If this is included, the
            function uses the survey history file to find instances of commas
            or semicolons in answer choices
    Returns:
        DataFrame with stacked data, a field for the beiwe ID, a field for the
        day of week.
    """
    audio_config_id_dict = get_config_id_dict(config_path)

    if time_end is None:
        time_end = get_month_from_today()
    if users is None:
        users = get_ids(download_folder)
    if len(users) == 0:
        logger.warning("No users found")
        return pd.DataFrame(columns=["Local time"], dtype="datetime64[ns]")

    all_users_list = [
        read_user_audio_recordings_stream(
            download_folder, user, tz_str, time_start, time_end, history_path
        )
        for user in users
    ]

    aggregated_data = pd.concat(all_users_list, axis=0, ignore_index=True)

    if aggregated_data.shape[0] == 0:
        logger.warning("No audio_recordings data found")
        return pd.DataFrame(columns=["Local time"], dtype="datetime64[ns]")

    aggregated_data["data_stream"] = "audio_recordings"

    aggregated_data["config_id"] = aggregated_data[
        "question text"
    ].apply(
        lambda x: audio_config_id_dict[x] if x in audio_config_id_dict.keys()
        else np.nan
    )

    return aggregated_data
