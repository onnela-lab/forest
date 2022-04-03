import datetime
import json
import logging
import os
import re

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
    "Android": {"Checkbox Question": "checkbox",
                "Info Text Box": "info_text_box",
                "Open Response Question": "free_response",
                "Radio Button Question": "radio_button",
                "Slider Question": "slider"},
    "iOS": {"checkbox": "checkbox",
            "free_response": "free_response",
            "info_text_box": "info_text_box",
            "radio_button": "radio_button",
            "slider": "slider"}
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
                            surv["answer_" + str(j)] = a["text"]

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
    df_merged = append_from_answers(df_merged, study_dir,
                                    participant_ids=users, tz_str=study_tz,
                                    config_path=config_path)

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
    agg_data = append_from_answers(agg_data, study_dir, tz_str=study_tz)

    return agg_data.reset_index(drop=True)


def append_from_answers(
        agg_data: pd.DataFrame, download_folder: str,
        participant_ids: list = None, tz_str: str = "UTC",
        config_path: str = None
) -> pd.DataFrame:
    answers_data = read_aggregate_answers_stream(
        download_folder, participant_ids, tz_str, config_path
    )
    if participant_ids is None:
        participant_ids = [u
                           for u in os.listdir(download_folder)
                           if not u.startswith(".") and u != "registry"]
    missing_submission_data = []  # list of surveys to add on to end

    for u in participant_ids:
        for survey_id in agg_data["survey id"].unique():
            known_timings_submits = agg_data.loc[
                (agg_data["beiwe_id"] == u)
                & (agg_data["survey id"] == survey_id),
                "Local time"
            ].unique()

            known_answers_submits = answers_data.loc[
                (answers_data["beiwe_id"] == u)
                & (answers_data["survey id"] == survey_id),
                "Local time"
            ].unique()
            missing_times = []
            for time in known_answers_submits:

                hours_from_nearest = np.min(
                    np.abs((pd.to_datetime(known_timings_submits)
                            - pd.to_datetime(time)).total_seconds())
                ) / 60/60
                # add on the data if there is more than 1/2 hour between an
                # answers submission and a timing submission.
                if hours_from_nearest > .5 or len(known_timings_submits) == 0:
                    missing_times.append(time)
            if len(missing_times) > 0:
                missing_data = answers_data.loc[
                        (answers_data["beiwe_id"] == u)
                        & (answers_data["survey id"] == survey_id)
                        & (answers_data["Local time"].isin(missing_times)),
                        :
                    ].copy()
                max_surv_inst_flg = answers_data.loc[
                        (answers_data["beiwe_id"] == u)
                        & (answers_data["survey id"] == survey_id)
                        & (answers_data["Local time"].isin(missing_times)),
                        "surv_inst_flg"
                ].max()
                # add the max submit flag to the inst_flags we have to make
                # sure all flags are unique
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
                missing_submission_data.append(missing_data)
    data_to_return = [agg_data] + missing_submission_data

    return pd.concat(data_to_return)


def read_one_answers_stream(download_folder: str, beiwe_id: str,
                            tz_str: str = "UTC") -> pd.DataFrame:
    """
    Reads in all answers data for a user and creates a column with the survey
    ID, as well as a column for the date from the filename.
    Args:
        download_folder (str):
            path to downloaded data. A folder wiith the user ID should be a
            subdirectory of this path.
        beiwe_id (str):
            ID of user to aggregate data
        tz_str (str):
            Time Zone to include in Local time column of output. See
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for
            options
    Returns:
        aggregated_data (DataFrame): dataframe with stacked data, a field for
        the beiwe ID, a field for the survey, and a filed with the time in
        the filename.
    """
    data_stream = "survey_answers"
    st_path = os.path.join(download_folder, beiwe_id, data_stream)
    if os.path.isdir(st_path):
        # get all survey IDs included for this user (data will have one folder
        # per survey)
        survey_ids = [subdir
                      for subdir in os.listdir(st_path)
                      if not subdir.startswith(".") and subdir != "registry"]
        all_surveys = []
        for survey in survey_ids:
            # get all csv files in the survey subdirectory
            all_files = [file
                         for file in os.listdir(os.path.join(st_path, survey))
                         if file.endswith(".csv")]
            survey_dfs = []
            # We need to enumerate to tell different survey occasions apart
            for i, file in enumerate(all_files):
                current_df = pd.read_csv(os.path.join(st_path, survey, file))
                # get UTC time from the file name because it"s not included in
                # the csv file
                file_name = file.split(".")[0]
                # the files are written with _ instead of :, so we need to
                # switch that back for format to read correctly.
                current_df["UTC time"] = pd.to_datetime(
                    file_name.replace("_", ":")
                )
                current_df["survey id"] = survey
                current_df["surv_inst_flg"] = i
                survey_dfs.append(current_df)
            survey_data = pd.concat(survey_dfs, axis=0, ignore_index=True)
            survey_data["beiwe_id"] = beiwe_id
            survey_data["Local time"] = survey_data[
                "UTC time"
            ].dt.tz_convert(tz_str).dt.tz_localize(None)

            all_surveys.append(survey_data)

        return pd.concat(all_surveys, axis=0, ignore_index=True)


def read_aggregate_answers_stream(
        download_folder: str, participant_ids: list = None,
        tz_str: str = "UTC", config_path: str = None
) -> pd.DataFrame:
    """
    Reads in all answers data for many users and fixes Android users to have
    an answer instead of an integer
    Args:
        download_folder (str):
            path to downloaded data. This folder should have Beiwe IDs as
            subdirectories.
        participant_ids (str):
            List of IDs of users to aggregate data on
        tz_str (str):
            Time Zone to include in Local time column of output. See
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for
            options
        config_path: Path to config file. If this is included, the function
            uses the config file to resolve semicolons that appear in survey
            answers lists. If this is not included, the function attempt to use
            iPhone responses to resolve semicolons.
    Returns:
        aggregated_data (DataFrame): dataframe with stacked data, a field for
        the beiwe ID, a field for the day of week.
    """
    if config_path is not None:
        config_surveys = parse_surveys(config_path, answers_l=True)
        config_included = True
    else:
        config_surveys = pd.DataFrame(None)
        config_included = False
    if participant_ids is None:
        participant_ids = [u
                           for u in os.listdir(download_folder)
                           if not u.startswith(".") and u != "registry"]

    all_users_list = [read_one_answers_stream(download_folder, user, tz_str)
                      for user in participant_ids]

    aggregated_data = pd.concat(all_users_list, axis=0, ignore_index=True)

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
        aggregated_data[
            "question type"] == "radio_button", "question answer options"
    ].unique()

    for answer_choices in radio_answer_choices_list:
        fixed_answer_choices = answer_choices
        if config_included:
            question_id = aggregated_data.loc[
                aggregated_data["question answer options"] == answer_choices,
                "question_id"
                ].unique()[0]
            answer_list = config_surveys.loc[
                config_surveys["question_id"] == question_id,
                config_surveys.columns[range(5, len(config_surveys.columns))]
            ].stack().unique()
            pass
        else:
            answer_list = aggregated_data.loc[
                aggregated_data[
                    "question answer options"] == answer_choices, "answer"
            ].unique()  # all answers users have put for this choice
        for answer in answer_list:
            if isinstance(answer, str) and answer.find(";") != -1:
                fixed_answer = answer.replace(";", ", ")
                # replace semicolons with commas within the answer. We include
                # a space after becaus we removed spaces after semicolons
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

    android_radio_questions = aggregated_data.loc[
        # find radio button questions. These are the ones that show up weird
        # on Android.
        (aggregated_data["question type"] == "Radio Button Question")
        # they will have ints in their answer field.
        & (aggregated_data["answer"].apply(
            lambda x: x.isdigit() if isinstance(x, str)
            else True if isinstance(x, int)
            else False)),
        "answer"
        # if their question type looks like this, it was made on an Android
        # device. Radio button questions are the only ones with the integer
        # instead of text (Also, avoiding any possible text outputs)
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
        QUESTION_TYPES_LOOKUP["Android"][x] if x in QUESTION_TYPES_LOOKUP[
            "Android"].keys()
        else x
    )
    return aggregated_data
