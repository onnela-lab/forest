import datetime
import json
import logging
import os

import glob
import numpy as np
import pandas as pd

from typing import Optional


logger = logging.getLogger(__name__)


def read_json(study_dir: str) -> dict:
    """Read a json file into a dictionary

    Args:
        study_dir (str):  study_dir to json file.
    Returns:
        dictionary (dict)
    """
    with open(study_dir, "r") as f:
        dictionary = json.load(f)
    return dictionary


# load events & question types dictionary
QUESTION_TYPES_LOOKUP = {
    'Android': {'Checkbox Question': 'checkbox',
                'Info Text Box': 'info_text_box',
                'Open Response Question': 'free_response',
                'Radio Button Question': 'radio_button',
                'Slider Question': 'slider'},
    'iOS': {'checkbox': 'checkbox',
            'free_response': 'free_response',
            'info_text_box': 'info_text_box',
            'radio_button': 'radio_button',
            'slider': 'slider'}
}


def q_types_standardize(q: str, lkp: Optional[dict] = None) -> str:
    """Standardizes question types using a lookup function

    Args:
        q (str):
            a single value for a question type
        lkp (dict):
            a lookup dictionary of question types and what they should map too.
            Based on Josh's dictionary of question types.

    Returns:
        s: string with the corrected question type
    """
    if lkp is None:
        lkp = QUESTION_TYPES_LOOKUP
    # If it's an Android string, flip it from the key to the value
    if q in lkp["Android"].keys():
        return lkp["Android"][q]
    else:
        return q


def read_and_aggregate(study_dir: str, beiwe_id: str,
                       data_stream: str) -> pd.DataFrame:
    """Read and aggregate data for a user

    Reads in all downloaded data for a particular user and data stream and
    stacks the datasets

    Args:
        study_dir (str):
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        beiwe_id (str):
            ID of user to aggregate data
        data_stream (str):
            Data stream to aggregate. Must be a datastream name as downloaded
            from the server (TODO: ADD A CHECK)

    Returns:
        survey_data (DataFrame): dataframe with stacked data, a field for the
        beiwe ID, a field for the day of week.
    """
    st_path = os.path.join(study_dir, beiwe_id, data_stream)
    if os.path.isdir(st_path):
        # get all survey timings files
        all_files = glob.glob(os.path.join(st_path, "*/*.csv"))
        # Sort file paths for when they're read in
        all_files = sorted(all_files)
        # Read in all files
        survey_data_list = [pd.read_csv(file) for file in all_files]
        survey_data: pd.DataFrame = pd.concat(survey_data_list,
                                              axis=0, ignore_index=False)
        survey_data["beiwe_id"] = beiwe_id
        survey_data["UTC time"] = survey_data["UTC time"].astype(
            "datetime64[ns]"
        )
        survey_data["DOW"] = survey_data["UTC time"].dt.dayofweek
        return survey_data
    else:
        logger.warning("No survey_timings for user %s.", beiwe_id)
        return pd.DataFrame(columns=["UTC time"], dtype="datetime64[ns]")


def aggregate_surveys(study_dir: str, users: list = None) -> pd.DataFrame:
    """Aggregate Survey Data

    Reads all survey data from a downloaded study folder and stacks data
    together. Standardizes question types between iOS and Android devices.

    Args:
        study_dir(str):
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        users(list):
            List of users to aggregate survey data over

    Returns:
        all_data(DataFrame): An aggregated dataframe that has a question index
        field to understand if there are multiple lines for one question.
    """
    # READ AND AGGREGATE DATA
    # get a list of users (ignoring hidden files and registry file downloaded
    # when using mano)
    if users is None:
        users = [u
                 for u in os.listdir(study_dir)
                 if not u.startswith(".") and u != "registry"]

    if len(users) == 0:
        logger.error("No users in directory %s", study_dir)
        return pd.DataFrame(columns=["UTC time"], dtype="datetime64[ns]")

    all_data_list = []
    for u in users:
        all_data_list.append(
            read_and_aggregate(study_dir, u, "survey_timings")
        )

    # Collapse all users into one file and drop duplicates
    all_data: pd.DataFrame = pd.concat(
        all_data_list, axis=0, ignore_index=False
    ).drop_duplicates().sort_values(["survey id", "beiwe_id", "timestamp"])

    # FIX EVENT FIELDS
    # Ensure there is an "event" field (There won't be one if all users are
    # Android)
    if "event" not in all_data.columns:
        all_data["event"] = None
    # Move Android events from the question id field to the event field
    all_data.event = all_data.apply(
        lambda row: row["question id"] if row["question id"] in [
            "Survey first rendered and displayed to user",
            "User hit submit"
        ] else row["event"],
        axis=1
    )
    all_data["question id"] = all_data.apply(
        lambda row: np.nan if row["question id"] == row["event"]
        else row["question id"],
        axis=1
    )

    # Fix question types
    all_data["question type"] = all_data.apply(
        lambda row: q_types_standardize(
            row["question type"], QUESTION_TYPES_LOOKUP
        ), axis=1
    )

    # ADD A QUESTION INDEX (to track changed answers)
    all_data["question id lag"] = all_data["question id"].shift(1)
    all_data["question index"] = all_data.apply(
        lambda row:
        1 if (row["question id"] != row["question id lag"]) else 0,
        axis=1
    )
    all_data["question index"] = all_data.groupby(["survey id", "beiwe_id"])[
        "question index"].cumsum()
    del all_data["question id lag"]

    # Add a survey instance ID that is tied to the submit line
    all_data["surv_inst_flg"] = 0
    all_data.loc[(all_data.event == "submitted") |
                 (all_data.event == "User hit submit") |
                 (all_data.event == "notified"), ["surv_inst_flg"]] = 1
    all_data["surv_inst_flg"] = all_data["surv_inst_flg"].shift(1)
    # If a change of survey occurs without a submit flg, flag the new line
    all_data["survey_prev"] = all_data["survey id"].shift(1)
    all_data.loc[
        all_data["survey_prev"] != all_data["survey id"], ["surv_inst_flg"]
    ] = 1
    del all_data["survey_prev"]

    # If "Survey first rendered and displayed to user", also considered a new
    # survey
    all_data.loc[
        all_data["event"] == "Survey first rendered and displayed to user",
        ["surv_inst_flg"]
    ] = 1

    # if a survey has a gap greater than 2 hours, consider it two surveys
    all_data["time_prev"] = all_data["UTC time"].shift(1)
    all_data["time_diff"] = all_data["UTC time"] - all_data["time_prev"]
    # Splitting up surveys where there appears to be a time gap or no submit
    # line.
    all_data.loc[
        all_data.time_diff > datetime.timedelta(hours=2), ["surv_inst_flg"]
    ] = 1

    all_data["surv_inst_flg"] = all_data.groupby(
        ["survey id", "beiwe_id"]
    )["surv_inst_flg"].cumsum()
    all_data.loc[all_data.surv_inst_flg.isna(), ["surv_inst_flg"]] = 0

    # OUTPUT AGGREGATE
    return all_data.reset_index(drop=True)


def parse_surveys(config_path: str, answers_l: bool = False) -> pd.DataFrame:
    """Get survey information from config path

    Args:
        config_path(str):
            path to the study configuration file
        answers_l(bool):
            If True, include question answers in summary

    Returns:
        A dataframe with all surveys, question ids, question texts,
        question types
    """
    data = read_json(config_path)
    surveys = data["surveys"]

    # Create an array for surveys and one for timings
    output = []

    for i, s in enumerate(surveys):
        # Pull out timings
        for q in s["content"]:
            if "question_id" in q.keys():
                surv = {
                        "config_id": i,
                        "question_id":  q["question_id"],
                        "question_text": q["question_text"],
                        "question_type":  q["question_type"]
                }
                if "text_field_type" in q.keys():
                    surv["text_field_type"] = q["text_field_type"]
                # Convert surv to data frame

                if answers_l:
                    if "answers" in q.keys():
                        for j, a in enumerate(q["answers"]):
                            surv["answer_" + str(i)] = a["text"]

                output.append(pd.DataFrame([surv]))
    output = pd.concat(output).reset_index(drop=True)
    return output


def convert_timezone_df(df_merged: pd.DataFrame, tz_str: str = "UTC",
                        utc_col: str = "UTC time") -> pd.DataFrame:
    """Convert a df to local time zone

    Args:
        df_merged(DataFrame):
            Dataframe that has a field of dates that are in UTC time
        tz_str(str):
            Study timezone (this should be a string from the pytz library)
        utc_col:
            Name of column in data that has UTC time dates

    Returns:
        df_merged(DataFrame):

    """

    df_merged["Local time"] = df_merged[utc_col].dt.tz_localize(
        "UTC"
    ).dt.tz_convert(tz_str).dt.tz_localize(None)

    return df_merged


def aggregate_surveys_config(
        study_dir: str, config_path: str, study_tz: str = "UTC",
        users: list = None
) -> pd.DataFrame:
    """Aggregate surveys when config is available

    Merges stacked survey data with processed configuration file data and
    removes lines that are not questions or submission lines

    Args:
        study_dir (str):
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        config_path(str):
            path to the study configuration file
        study_tz(str):
            Timezone of study. This defaults to "UTC"
        users(tuple):
            List of beiwe IDs of users to aggregate

    Returns:
        df_merged(DataFrame): Merged data frame
    """
    # Read in aggregated data and survey configuration
    config_surveys = parse_surveys(config_path)
    agg_data = aggregate_surveys(study_dir, users)
    if agg_data.shape[0] == 0:
        return agg_data

    # Merge data together and add configuration survey ID to all lines
    df_merged = agg_data.merge(
        config_surveys[["config_id", "question_id"]], how="left",
        left_on="question id", right_on="question_id"
    ).drop(["question_id"], axis=1)
    df_merged["config_id_update"] = df_merged["config_id"].fillna(
        method="ffill"
    )
    df_merged["config_id"] = df_merged.apply(
        lambda row:
        row["config_id_update"] if row["event"] in ["User hit submit",
                                                    "submitted"]
        else row["config_id"],
        axis=1
    )

    del df_merged["config_id_update"]

    # Mark submission lines
    df_merged["submit_line"] = df_merged.apply(
        lambda row:
        1 if row["event"] in ["User hit submit", "submitted"] else 0,
        axis=1
    )

    # Remove notification and expiration lines
    df_merged = df_merged.loc[(~df_merged["question id"].isnull())
                              | (~df_merged["config_id"].isnull())]

    # Convert to the study's timezone
    df_merged = convert_timezone_df(df_merged, study_tz)

    return df_merged.reset_index(drop=True)


def aggregate_surveys_no_config(study_dir: str, study_tz: str = "UTC",
                                users: list = None) -> pd.DataFrame:
    """Clean aggregated data

    Args:
        study_dir (str):
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        study_tz(str):
            Timezone of study. This defaults to "UTC"
        users(tuple):
            List of Beiwe IDs to run

    Returns:
        df_merged(DataFrame): Merged data frame
    """
    agg_data = aggregate_surveys(study_dir, users)
    if agg_data.shape[0] == 0:
        return agg_data
    agg_data["submit_line"] = agg_data.apply(
        lambda row:
        1 if row["event"] in ["User hit submit", "submitted"] else 0,
        axis=1
    )

    # Remove notification and expiration lines
    agg_data = agg_data.loc[(~agg_data["question id"].isnull())]

    # Convert to the study's timezone
    agg_data = convert_timezone_df(agg_data, tz_str=study_tz)

    return agg_data.reset_index(drop=True)
