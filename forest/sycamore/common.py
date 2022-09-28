import datetime
import logging
import os
import re
from typing import Optional, Dict

import glob
import numpy as np
import pandas as pd

from forest.utils import get_ids
from forest.sycamore.constants import (EARLIEST_DATE, QUESTION_TYPES_LOOKUP,
                                       ANDROID_NULLABLE_ANSWER_CHANGE_DATE)
from forest.sycamore.read_audio import read_aggregate_audio_recordings_stream
from forest.sycamore.utils import (read_json, get_month_from_today,
                                   filename_to_timestamp)


logger = logging.getLogger(__name__)


def safe_read_csv(filepath: str) -> pd.DataFrame:
    """Read a csv file, returning an empty dataframe if the file is corrupted

    Args:
        filepath: The filepath to read in

    Returns:
        A pandas DataFrame with information in the csv file (if the file is
        formatted correctly), or a blank pandas DataFrame.
    """
    try:
        return pd.read_csv(filepath)
    except UnicodeDecodeError:
        # If a file is corrupted, don't bother reading it in.
        logger.error("Unicode Error When Reading %s", filepath)
        return pd.DataFrame()
    except pd.errors.ParserError:
        # another way the file can be corrupted
        logger.error("Parser Error When Reading %s", filepath)
        return pd.DataFrame()


def q_types_standardize(q: str, lkp: Optional[dict] = None) -> str:
    """Standardizes question types using a lookup function

    Args:
        q (str):
            a single value for a question type
        lkp (dict):
            a lookup dictionary of question types and what they should map too.
            Based on Josh's dictionary of question types.

    Returns:
        string with the corrected question type
    """
    if lkp is None:
        lkp = QUESTION_TYPES_LOOKUP
    # If it's an Android string, flip it from the key to the value
    if q in lkp["Android"].keys():
        return lkp["Android"][q]
    else:
        return q


def read_and_aggregate(
        study_dir: str, user: str, data_stream: str,
        time_start: str = EARLIEST_DATE,
        time_end: str = None,
        tz_str: str = "UTC"
) -> pd.DataFrame:
    """Read and aggregate data for a user

    Reads in all survey_timings data for a particular user and data stream and
    stacks the datasets

    Args:
        study_dir (str):
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        user (str):
            ID of user to aggregate data
        data_stream (str):
            Data stream to aggregate. Must be a datastream name as downloaded
            from the server (TODO: ADD A CHECK)
        time_start(str):
            The first date of the survey data, in YYYY-MM-DD format
        time_end(str):
            The last date of the survey data, in YYYY-MM-DD format
        tz_str(str):
            Time zone corresponding to time_start and time_end

    Returns:
        Dataframe with stacked data, a field for the beiwe ID, a field for the
        day of week.
    """
    if time_end is None:
        time_end = get_month_from_today()
    st_path = os.path.join(study_dir, user, data_stream)
    if os.path.isdir(st_path):
        # get all survey timings files
        all_files = glob.glob(os.path.join(st_path, "*/*.csv"))
        # Sort file paths for when they're read in
        all_files = sorted(all_files)
        # Read in all files
        timestamp_start = pd.to_datetime(time_start)
        timestamp_end = pd.to_datetime(time_end)
        survey_data_list = []
        for file in all_files:
            filename = os.path.basename(file)
            if (timestamp_start < filename_to_timestamp(filename, tz_str)
                    < timestamp_end):
                survey_data_list.append(safe_read_csv(file))
        if len(survey_data_list) == 0:
            logger.warning("No survey_timings for user %s.", user)
            return pd.DataFrame(columns=["UTC time"], dtype="datetime64[ns]")

        survey_data: pd.DataFrame = pd.concat(survey_data_list,
                                              axis=0, ignore_index=False)
        survey_data["beiwe_id"] = user
        survey_data["UTC time"] = survey_data["UTC time"].astype(
            "datetime64[ns]"
        )
        survey_data["DOW"] = survey_data["UTC time"].dt.dayofweek
        return survey_data
    else:
        logger.warning("No survey_timings for user %s.", user)
        return pd.DataFrame(columns=["UTC time"], dtype="datetime64[ns]")


def aggregate_surveys(
        study_dir: str, users: list = None,
        time_start: str = EARLIEST_DATE,
        time_end: str = None, tz_str: str = "UTC"
) -> pd.DataFrame:
    """Aggregate Survey Data

    Reads all survey_timings data from a downloaded study folder and stacks
    data together. Standardizes question types between iOS and Android devices.

    Args:
        study_dir(str):
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        users(list):
            List of users to aggregate survey data over
        time_start(str):
            The first date of the survey data, in YYYY-MM-DD format
        time_end(str):
            The last date of the survey data, in YYYY-MM-DD format
        tz_str(str):
            The time zone corresponding to time_start and time_end

    Returns:
        A DataFrame that has a question index field to understand if there are
        multiple lines for one question.
    """
    # READ AND AGGREGATE DATA
    # get a list of users (ignoring hidden files and registry file downloaded
    # when using mano)
    if users is None:
        users = get_ids(study_dir)
    if time_end is None:
        time_end = get_month_from_today()

    if len(users) == 0:
        logger.error("No users in directory %s", study_dir)
        return pd.DataFrame(columns=["UTC time"], dtype="datetime64[ns]")

    all_data_list = []
    for user in users:
        all_data_list.append(
            read_and_aggregate(study_dir, user, "survey_timings", time_start,
                               time_end, tz_str)
        )

    # Collapse all users into one file and drop duplicates
    all_data: pd.DataFrame = pd.concat(
        all_data_list, axis=0, ignore_index=False
    )

    if all_data.shape[0] == 0:
        logger.error("No data in directory %s in given time frame", study_dir)
        return pd.DataFrame(columns=["UTC time"], dtype="datetime64[ns]")

    all_data = all_data.drop_duplicates().sort_values(
        ["survey id", "beiwe_id", "timestamp"]
    )

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
    # Replace the question ID in these rows with a valid question ID for the
    # survey so that survey scheduling metrics can incorporate surveys that
    # were opened.
    # If the only other rows in all_data were rows describing that the survey
    # was first rendered, put "" for the question ID.
    all_data["question id"] = all_data.apply(
        lambda row:
        (all_data.loc[(all_data["survey id"] == row["survey id"]) &
                       (all_data["question id"] != all_data["event"]),
                       "question id"].tolist() + [""])[0]
        if row["question id"] == row["event"]
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
        A DataFrame with all surveys, question ids, question texts,
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
                            surv["answer_" + str(j)] = a["text"]
                output.append(pd.DataFrame([surv]))
    if len(output) == 0:
        logger.warning("No Data Found")
        return pd.DataFrame()
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
        DataFrame with a column "Local time" with the time in local time

    """

    df_merged["Local time"] = df_merged[utc_col].dt.tz_localize(
        "UTC"
    ).dt.tz_convert(tz_str).dt.tz_localize(None)

    return df_merged


def aggregate_surveys_config(
        study_dir: str, config_path: str, study_tz: str = "UTC",
        users: list = None, time_start: str = EARLIEST_DATE,
        time_end: str = None, augment_with_answers: bool = True,
        history_path: str = None, include_audio_surveys: bool = True
) -> pd.DataFrame:
    """Aggregate surveys when config is available

    Merges stacked survey data with processed configuration file data and
    removes lines that are not questions or submission lines. Uses
    survey_answers to fill in missing survey_timings data when needed.

    Args:
        study_dir:
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        config_path:
            path to the study configuration file
        study_tz:
            Timezone of study. This defaults to "UTC"
        users:
            List of beiwe IDs of users to aggregate
        augment_with_answers:
            Whether to use the survey_answers stream to fill in missing surveys
            from survey_timings
        time_start:
            The first date of the survey data, in YYYY-MM-DD format
        time_end:
            The last date of the survey data, in YYYY-MM-DD format
        history_path:
            Path to survey history file. If this is included, the
            survey history file is used to find instances of commas or
            semicolons in answer choices to determine the correct choice for
            Android radio questions
        include_audio_surveys:
            Whether to include submissions of audio surveys in addition to text
            surveys

    Returns:
        DataFrame of questions and submission lines.
    """
    if time_end is None:
        time_end = get_month_from_today()
    # Read in aggregated data and survey configuration
    config_surveys = parse_surveys(config_path)
    agg_data = aggregate_surveys(study_dir, users, time_start, time_end,
                                 study_tz)
    if agg_data.shape[0] == 0:
        return agg_data

    # Merge data together and add configuration survey ID to all lines
    # Pandas gives an error if question IDs are different data types.
    # So, we will convert everything to string.
    # But, if we just do raw conversion to string, we run into problems where
    # it tries to merge on 'nan'. So, we will convert all of the 'nan' to None
    # before merging.
    agg_data["question id"] = np.where(
        (agg_data["question id"].astype(str) == "nan").to_numpy(),
        np.full(agg_data.shape[0], None), agg_data["question id"].astype(str)
    )
    config_surveys["question_id"] = np.where(
        (config_surveys["question_id"].astype(str) == "nan").to_numpy(),
        np.full(config_surveys.shape[0], None),
        config_surveys["question_id"].astype(str)
    )
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
    df_merged = fix_radio_answer_choices(df_merged, config_path, history_path)
    if augment_with_answers:
        df_merged["data_stream"] = "survey_timings"
        df_merged = append_from_answers(df_merged, study_dir, users,
                                        study_tz, time_start, time_end,
                                        config_path, history_path)
    if include_audio_surveys:
        audio_surveys = read_aggregate_audio_recordings_stream(
            study_dir, users, study_tz, config_path, time_start, time_end,
            history_path
        )
        df_merged = pd.concat(
            [df_merged, audio_surveys], axis=0, ignore_index=False
        )

    return df_merged.reset_index(drop=True)


def aggregate_surveys_no_config(
        study_dir: str, study_tz: str = "UTC", users: list = None,
        time_start: str = EARLIEST_DATE, time_end: str = None,
        augment_with_answers: bool = True, include_audio_surveys: bool = True
) -> pd.DataFrame:
    """Clean aggregated data

    Args:
        study_dir (str):
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        study_tz(str):
            Timezone of study. This defaults to "UTC"
        users(tuple):
            List of Beiwe IDs to run
        time_start(str):
            The first date of the survey data, in YYYY-MM-DD format
        time_end(str):
            The last date of the survey data, in YYYY-MM-DD format
        augment_with_answers(bool):
            Whether to use the survey_answers stream to fill in missing surveys
            from survey_timings
        include_audio_surveys:
            Whether to include submissions of audio surveys in addition to text
            surveys. Default is True

    Returns:
        DataFrame of questions and submission lines
    """
    if time_end is None:
        time_end = get_month_from_today()
    agg_data = aggregate_surveys(study_dir, users, time_start, time_end,
                                 study_tz)
    if agg_data.shape[0] == 0:
        return agg_data
    agg_data["submit_line"] = agg_data.apply(
        lambda row:
        1 if row["event"] in ["User hit submit", "submitted"] else 0,
        axis=1
    )

    # Remove lines where the event is 'notified' or 'expired'. These lines
    # will show up on file collected from iOS devices, and they happen when a
    # user gets notified of a survey and the survey expires because another
    # survey gets sent.
    agg_data = agg_data.loc[(~agg_data["question id"].isnull())]

    # Convert to the study's timezone
    agg_data = convert_timezone_df(agg_data, tz_str=study_tz)
    agg_data = fix_radio_answer_choices(agg_data)
    if augment_with_answers:
        agg_data["data_stream"] = "survey_timings"
        agg_data = append_from_answers(agg_data, study_dir, users,
                                       study_tz, time_start, time_end)
    if include_audio_surveys:
        audio_surveys = read_aggregate_audio_recordings_stream(
            study_dir, users, study_tz, None, time_start, time_end,
            None
        )
        agg_data = pd.concat(
            [agg_data, audio_surveys], axis=0, ignore_index=False
        )

    return agg_data.reset_index(drop=True)


def append_from_answers(
        agg_data: pd.DataFrame, download_folder: str,
        users: list = None, tz_str: str = "UTC",
        time_start: str = EARLIEST_DATE, time_end: str = None,
        config_path: str = None, history_path: str = None
) -> pd.DataFrame:
    """Append surveys included in survey_answers to data from survey_timings.

    Sometimes, a survey is missing from survey_timings. If survey_answers
    contains a survey that is missing from survey_timings, this function adds
    the less complete information from survey_answers to fill in the gaps.

    Args:
        agg_data: Dataframe with aggregated data (output from
            aggregate_surveys_config)
        download_folder:
            path to downloaded data. This is a folder that includes the user
            data in a subfolder with the beiwe_id as the subfolder name
        tz_str:
            Timezone to use for "Local time" column values. This defaults to
            "UTC"
        users:
            List of Beiwe IDs used to augment with data
        time_start:
            The first date of the survey data, in YYYY-MM-DD format
        time_end:
            The last date of the survey data, in YYYY-MM-DD format
        config_path:
            Filepath to survey config file (downloaded from Beiwe website)
        history_path:
            Path to survey history file. If this is included, the
            survey history file is used to find instances of commas or
            semicolons in answer choices to determine the correct choice for
            Android radio questions

    Returns:
        Data frame with all survey_timings data, including data from
        survey_answers where survey_timings data is missing.
    """
    if time_end is None:
        time_end = get_month_from_today()
    answers_data = read_aggregate_answers_stream(
        download_folder, users, tz_str, config_path, time_start,
        time_end, history_path
    )
    if answers_data.shape[0] == 0:
        return agg_data

    if users is None:
        users = get_ids(download_folder)
    missing_submission_data = []  # list of surveys to add on to end

    for user in users:
        for survey_id in agg_data["survey id"].unique():
            missing_data = find_missing_data(user, survey_id, agg_data,
                                             answers_data)
            if missing_data.shape[0] == 0:
                continue
            missing_data["question index"] += \
                np.max(agg_data["question index"]) + 1
            missing_submission_data.append(missing_data)
    return pd.concat([agg_data] + missing_submission_data
                     ).sort_values("UTC time")


def find_missing_data(user: str, survey_id: str, agg_data: pd.DataFrame,
                      answers_data: pd.DataFrame) -> pd.DataFrame:
    """Get data present in answers_data that is missing from agg_Data

    Args:
        user: String with the Beiwe ID to look for missing data
        survey_id: String with the survey ID to look for missing data in
        agg_data: Dataframe with aggregated timings data (output from
            aggregate_surveys_config)
        answers_data: Dataframe with aggregated answers data (output from
            read_aggregate_answers_stream)

    Returns:
        A DataFrame with any survey data that is present in
            answers_data but missing in agg_data

    """
    known_timings_submits = agg_data.loc[
        (agg_data["beiwe_id"] == user)
        & (agg_data["survey id"] == survey_id),
        "Local time"
    ].unique()

    known_answers_submits = answers_data.loc[
        (answers_data["beiwe_id"] == user)
        & (answers_data["survey id"] == survey_id),
        "Local time"
    ].unique()
    missing_times = []
    for time in known_answers_submits:

        hours_from_nearest = np.min(
            np.abs((pd.to_datetime(known_timings_submits)
                    - pd.to_datetime(time)).total_seconds())
        ) / 60 / 60
        # add on the data if there is more than 1/2 hour between an
        # answers submission and a timing submission.
        if hours_from_nearest > .5 or len(known_timings_submits) == 0:
            missing_times.append(time)
    if len(missing_times) > 0:
        missing_data = answers_data.loc[
                       (answers_data["beiwe_id"] == user)
                       & (answers_data["survey id"] == survey_id)
                       & (answers_data["Local time"].isin(missing_times)),
                       :
                       ].copy()
        # Get the max survey flag from agg_data so we don't overlap any
        # survey flags after merging
        max_surv_inst_flg = agg_data.loc[
            (agg_data["beiwe_id"] == user)
            & (agg_data["survey id"] == survey_id),
            "surv_inst_flg"
        ].max()
        if np.isnan(max_surv_inst_flg):
            max_surv_inst_flg = 0
        # add the max submit flag to the inst_flags we have
        missing_data["surv_inst_flg"] = missing_data[
                                            "surv_inst_flg"
                                        ] + max_surv_inst_flg + 1

        missing_data["time_prev"] = missing_data["Local time"].shift(1)
        # one line in the survey will have a submission flag
        missing_data.loc[
            missing_data["time_prev"] != missing_data["Local time"],
            "submit_flg"
        ] = 1

        missing_data.drop(["time_prev"], axis=1, inplace=True)
        return missing_data
    else:
        return pd.DataFrame()


def read_user_answers_stream(
        download_folder: str, user: str, tz_str: str = "UTC",
        time_start: str = EARLIEST_DATE, time_end: str = None
) -> pd.DataFrame:
    """Reads in all survey_answers data for a user

    Reads survey_answers data and creates a column with the survey
    ID, as well as a column for the date from the filename.

    Args:
        download_folder (str):
            path to downloaded data. A folder wiith the user ID should be a
            subdirectory of this path.
        user (str):
            ID of user to aggregate data
        tz_str (str):
            Time Zone to include in Local time column of output. See
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for
            options
        time_start(str):
            The first date of the survey data, in YYYY-MM-DD format
        time_end(str):
            The last date of the survey data, in YYYY-MM-DD format

    Returns:
        DataFrame with stacked data, a field for the beiwe ID, a field for the
        survey, and a filed with the time in the filename.
    """
    if time_end is None:
        time_end = get_month_from_today()
    ans_dir = os.path.join(download_folder, user, "survey_answers")
    if os.path.isdir(ans_dir):
        # get all survey IDs included for this user (data will have one folder
        # per survey)
        survey_ids = get_ids(ans_dir)
        all_surveys = []
        timestamp_start = pd.to_datetime(time_start)
        timestamp_end = pd.to_datetime(time_end)
        for survey in survey_ids:
            # get all csv files in the survey subdirectory
            all_files = []
            for filepath in os.listdir(os.path.join(ans_dir, survey)):
                filename = os.path.basename(filepath)
                valid_file = (filepath.endswith(".csv")
                              and (timestamp_start
                                   < filename_to_timestamp(filename, tz_str)
                                   < timestamp_end))
                if valid_file:
                    all_files.append(filepath)

            if len(all_files) == 0:
                logger.warning("No survey_answers for user %s in given time "
                               "frames.", user)
                return pd.DataFrame(columns=["Local time"],
                                    dtype="datetime64[ns]")

            survey_dfs = []
            # We need to enumerate to tell different survey occasions apart
            for i, file in enumerate(all_files):
                current_df = safe_read_csv(os.path.join(ans_dir, survey, file))
                if current_df.shape[0] == 0:
                    continue
                # Add a submission line if they at least saw all of the
                # questions
                current_df["submit_line"] = 0
                if not (current_df["answer"] == "NOT_PRESENTED").any():
                    current_df = pd.concat([
                        current_df,
                        pd.DataFrame({"submit_line": [1], "answer": ""})
                    ]).reset_index()
                # Now, add column values that should apply to all rows
                current_df["survey id"] = survey
                filename = os.path.basename(file)
                current_df["UTC time"] = filename_to_timestamp(filename, "UTC")
                current_df["surv_inst_flg"] = i

                survey_dfs.append(current_df)
            if len(survey_dfs) == 0:
                logger.warning("No survey_answers for user %s.", user)
                return pd.DataFrame(columns=["UTC time"],
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


def read_aggregate_answers_stream(
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
    if time_end is None:
        time_end = get_month_from_today()
    if users is None:
        users = get_ids(download_folder)
    if len(users) == 0:
        logger.warning("No users found")
        return pd.DataFrame(columns=["Local time"], dtype="datetime64[ns]")

    all_users_list = [
        read_user_answers_stream(download_folder, user, tz_str, time_start,
                                 time_end)
        for user in users
    ]

    aggregated_data = pd.concat(all_users_list, axis=0, ignore_index=True)

    if aggregated_data.shape[0] == 0:
        logger.warning("No survey_answers data found")
        return pd.DataFrame(columns=["Local time"], dtype="datetime64[ns]")

    aggregated_data = fix_radio_answer_choices(aggregated_data, config_path,
                                               history_path)

    # Now, we will locate the indices of Android radio button questions that
    # have integers instead of strings.

    # First, android questions will have the android name for question type.
    android_radio_rows = (aggregated_data["question type"] ==
                          "Radio Button Question")
    # Next, any rows before this date will be strings, and any after will be
    # integers. So, we will only use rows that were created after the app
    # change happened.
    rows_after_nullable_change = (aggregated_data["UTC time"] >
                                  ANDROID_NULLABLE_ANSWER_CHANGE_DATE)
    # Finally, we will look at rows that include digits. Some people may have
    # downloaded the app before the nullable date and continued to have string
    # answers after the date because they didn't update the app, so we will
    # filter out any answers that had strings.
    rows_with_integer_answers = (aggregated_data["answer"].apply(
            lambda x: str(x).isdigit()))

    android_radio_questions = aggregated_data.loc[
        android_radio_rows & rows_after_nullable_change &
        rows_with_integer_answers, :
    ].index

    for i in android_radio_questions:
        # for Androids, the radio buttons are the index of the "question answer
        # options" list that the person selected. We will iterate through and
        # get the string corresponding to that index
        aggregated_data.loc[i, "answer"] = aggregated_data.loc[
            i, "question answer options"
        ][int(aggregated_data.loc[i, "answer"])]
    # convert to the iOS question types text
    aggregated_data["question type"] = aggregated_data["question type"].apply(
        lambda x:
        QUESTION_TYPES_LOOKUP["Android"][x]
        if x in QUESTION_TYPES_LOOKUP["Android"].keys()
        else x
    )

    aggregated_data["data_stream"] = "survey_answers"
    # Excluding rows without survey answers to be consistent with exclusions
    # in aggregating survey_timings
    if config_path is None:
        return aggregated_data.loc[
            aggregated_data["answer"] != "NOT_PRESENTED", :
        ]

    config_surveys = parse_surveys(config_path)
    # Merge data together and add configuration survey ID to all lines
    df_merged = aggregated_data.merge(
        config_surveys[["config_id", "question_id"]], how="left",
        left_on="question id", right_on="question_id"
    ).drop(["question_id"], axis=1)

    return df_merged.loc[
        df_merged["answer"] != "NOT_PRESENTED", :
    ]


def fix_radio_answer_choices(
        aggregated_data: pd.DataFrame, config_path: str = None,
        history_path: str = None
) -> pd.DataFrame:
    """
    Change the "question answer options" column into a list of question answer
        options. Also, correct for the fact that a semicolon may be a delimiter
        between choices, an actual semicolon in a question, or a sanitized ","

    Args:
        aggregated_data:
            Output from read_user_answers_stream or
            aggregate_surveys
        config_path:
            Path to config file. If this is included, the function
            uses the config file to resolve semicolons that appear in survey
            answers lists. If this is not included, the function attempt to use
            iPhone responses to resolve semicolons.
        history_path:
            Path to survey history file. If this is included, the
            function uses the survey history file to find instances of commas
            or semicolons in answer choices
    """
    # if a semicolon appears in an answer choice option,
    # our regexp sub/split operation would think there are
    # way more answers than there really are.
    # We will pull from the responses from iPhone users and switch semicolons
    # within an answer to commas.
    # Android users have "; " separating answer choices. iPhone users have
    # ";" separating choices.
    # This will make them the same.
    aggregated_data["question answer options"] = aggregated_data[
        "question answer options"].apply(
        lambda x: x.replace("; ", ";") if isinstance(x, str) else x
    )

    # We also need to change the answers to match the above pattern if we want
    # to find answers inside of answer choice strings.
    # We will change this back later.
    aggregated_data["answer"] = aggregated_data["answer"].apply(
        lambda x: x.replace("; ", ";") if isinstance(x, str) else x
    )

    # get all answer choices text options now that Android and iPhone have the
    # same ones.
    radio_answer_choices_list = aggregated_data.loc[
        aggregated_data["question type"].apply(
            lambda x: x in ["radio_button", "Radio Button Question"]
        ), "question answer options"
    ].unique()
    if config_path is not None:
        sep_dict = get_choices_with_sep_values(config_path, history_path)
    else:
        sep_dict = dict()

    for answer_choices in radio_answer_choices_list:
        fixed_answer_choices = answer_choices
        if config_path is not None:
            question_id = aggregated_data.loc[
                aggregated_data["question answer options"] == answer_choices,
                "question id"
            ].unique()[0]
            # sep_dict wil only include keys for question IDs with
            # semicolons/commas inside the answer choice, so we can skip that
            # question if it's not a key of sep_dict
            if question_id not in sep_dict.keys():
                continue
            answer_list = sep_dict[question_id]
        else:
            answer_list = aggregated_data.loc[
                aggregated_data[
                    "question answer options"] == answer_choices, "answer"
            ].unique()  # all answers users have put for this choice
        for answer in answer_list:
            if isinstance(answer, str) and answer.find(";") != -1:
                fixed_answer = answer.replace(";", ", ")
                # replace semicolons with commas within the answer. We include
                # a space after because we removed spaces after semicolons
                # earlier.
                fixed_answer_choices = fixed_answer_choices.replace(
                    answer, fixed_answer
                )  # the answer within the answer choice string no
                # longer has a semicolon.

        aggregated_data.loc[
            aggregated_data["question answer options"] == answer_choices,
            "question answer options"
        ] = fixed_answer_choices

    aggregated_data["answer"] = aggregated_data["answer"].apply(
        lambda x: x.replace(";", ", ") if isinstance(x, str) else x
    )

    # remove first and last bracket as well as any nan, then split it by ;
    aggregated_data["question answer options"] = aggregated_data[
        "question answer options"
    ].astype("str").apply(lambda x: re.sub(r"^\[|\]$|^nan$", "", x).split(";"))

    return aggregated_data


def update_qs_with_seps(qs_with_seps: dict, survey_content: dict) -> dict:
    """
    Iterates through answers in question_dict and adds any choices with , or ;
        to the correct entry in the sep_choices dict.

    Args:
        qs_with_seps: Dictionary with a key for each question ID. For each
            question ID, there is a set of all response choices with a , or ;
        survey_content: Dictionary with survey content information, from either
            the study config file or survey history file
    """
    for question in range(len(survey_content)):
        if "question_type" not in survey_content[question].keys():
            continue
        if survey_content[question]["question_type"] == "radio_button":
            question_dict = survey_content[question]
            answer_choices = question_dict["answers"]
            q_sep_choices = set()
            for choice in range(len(answer_choices)):
                answer_text = answer_choices[choice]["text"]
                if len(re.findall(",|;", answer_text)) != 0:
                    # At least one separation value occurs in the
                    # response
                    q_sep_choices.add(
                        answer_text.replace(",", ";").replace("; ", ";")
                    )
            if len(q_sep_choices) != 0:
                question_id = survey_content[question]["question_id"]
                if question_id in qs_with_seps.keys():
                    qs_with_seps[question_id] = qs_with_seps[
                        question_id
                    ].union(q_sep_choices)
                else:
                    qs_with_seps[question_id] = q_sep_choices
    return qs_with_seps


def get_choices_with_sep_values(config_path: str = None,
                                survey_history_path: str = None) -> dict:
    """
    Create a dict with a key for every question ID and a set of any responses
    for that ID that had a comma in them.

    Question IDs are included in the dict if they satisfy two conditions:
    1. They are radio button questions
    2. at least one of the response choices contains a semicolon or a comma

    Args:
        config_path:
            Path to config file. If this is included, the function
            uses the config file to resolve semicolons that appear in survey
            answers lists. If this is not included, the function attempt to use
            iPhone responses to resolve semicolons.
        survey_history_path:
            Path to survey history file. If this is included,
            the function uses the survey history file to find instances of
            commas or semicolons in answer choices


    """
    qs_with_seps: Dict[str, set] = {}
    if config_path is not None:
        study_config = read_json(config_path)
        if "surveys" in study_config.keys():
            surveys_list = study_config["surveys"]
        else:
            logger.warning("No survey information found in config file")
            return qs_with_seps
        for survey_num in range(len(surveys_list)):
            survey = surveys_list[survey_num]["content"]
            qs_with_seps = update_qs_with_seps(qs_with_seps, survey)
    if survey_history_path is not None:
        survey_history_dict = read_json(survey_history_path)
        for survey_id in survey_history_dict.keys():
            for version in range(len(survey_history_dict[survey_id])):
                survey = survey_history_dict[survey_id][version]["survey_json"]
                qs_with_seps = update_qs_with_seps(qs_with_seps, survey)
    else:
        logger.warning(
            "No survey history path included. If you have changed radio survey"
            " answer choices since starting your study, and if you used "
            "semicolons or commas in those answer choices, incorrect survey "
            "responses may be output for android devices"
        )
    return qs_with_seps


def write_data_by_user(df_to_write: pd.DataFrame, output_folder: str,
                       users: list = None):
    """
    Write a dataframe to csv files, with a csv file corresponding to each user.

    This function is used to mimic how files are written by
        forest.jasmine.gps_stats_main and forest.willow.log_stats_main

    Args:
        output_folder: path to folder to write csv files in
        df_to_write: dataframe to be written
        users: list of users to split dataframe by

    """
    os.makedirs(output_folder, exist_ok=True)

    if users is None:
        users = df_to_write.beiwe_id.unique().tolist()
    for user in users:
        current_df = df_to_write.loc[df_to_write.beiwe_id == user, :].copy()
        if current_df.shape[0] == 0:
            continue
        current_df.drop("beiwe_id", axis=1, inplace=True)
        path_to_write = os.path.join(output_folder, user + ".csv")
        current_df.to_csv(path_to_write, index=False)
