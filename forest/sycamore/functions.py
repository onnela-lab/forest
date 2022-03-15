import datetime
import json
import logging
import os

import glob
import numpy as np
import pandas as pd

from typing import Optional

# Explore use of logging function (TO DO: Read wiki)
logger = logging.getLogger(__name__)

# Modified from legacy beiwetools code


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
# From Josh"s legacy script
this_dir = os.path.dirname(__file__)
events = read_json(os.path.join(this_dir, "events.json"))
question_type_names = read_json(
    os.path.join(this_dir, "question_type_names.json")
)


def make_lookup() -> dict:
    """From legacy script

    Reformats the question types JSON to be usable in future functions
    """
    lookup: dict = {"iOS": {}, "Android": {}}
    for k in question_type_names:
        for opsys in ["iOS", "Android"]:
            opsys_name = question_type_names[k][opsys]
            lookup[opsys][opsys_name] = k
    return lookup


# Create a lookup to be used in question standardization
question_types_lookup = make_lookup()


def q_types_standardize(q: str, lkp: Optional[dict] = None) -> str:
    """Standardizes question types using a lookup function

    Args:
        q (str):
            a single value for a question type
        lkp (dict):
            a lookup dictionary of question types and what they should map too.
            Based on Josh"s dictionary of question types.

    Returns:
        s: string with the corrected question type
    """
    if lkp is None:
        lkp = question_types_lookup
    # If it"s an Android string, flip it from the key to the value
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
        # Sort file paths for when they"re read in
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
        logger.warning("No survey_timings for user %s." % beiwe_id)
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
        logger.error(f"No users in directory {study_dir}")
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
    # Ensure there is an "event" field (They"re won"t be one if all users are
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
            row["question type"], question_types_lookup
        ), axis=1
    )

    # ADD A QUESTION INDEX (to track changed answers)
    all_data["question id lag"] = all_data["question id"].shift(1)
    all_data["question index"] = all_data.apply(
        lambda row:
        1 if ((row["question id"] != row["question id lag"])) else 0,
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

    # if a survey has a gap greater than 5 hours, consider it two surveys
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
        # Pull out questions

        # Pull out timings
        #         timings = parse_timings(s, i)
        for q in s["content"]:
            if "question_id" in q.keys():
                surv = {}
                surv["config_id"] = i
                surv["question_id"] = q["question_id"]
                surv["question_text"] = q["question_text"]
                surv["question_type"] = q["question_type"]
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


def get_survey_timings(person_ids: list, study_dir: str,
                       survey_id: str) -> pd.DataFrame:
    """Get survey administration times

    Created on Thu Jan 28 11:34:23 2021

    @author: DEBEU

    Parameters
    ----------
    person_ids : list of beiwe_ids to .
    study_dir : raw data directory containing directories for each beiwe_id.
    survey_id : back_end id for survey (and name of folder within
    study_dir/beiwe_id/survey_timings).


    Returns
    -------
    Record with beiwe_id / phone_os / date_hour / start_time / end_time.
    For iOS users:
        start_time = time of "present" of first question
        end_time = time of "submitted"
    For Android
        start_time = ....
        end_time = ....

    Assumes
    -------
    Operating system-specific difference in registration of survey timings
    That a survey with survey_id is not triggered more than once an hour


    """
    record = np.array(
        ["beiwe_id", "phone_os", "date_hour", "start_time", "end_time"]
    ).reshape(1, 5)
    for pid in person_ids:
        survey_dir = os.path.join(study_dir, pid, "survey_timings", survey_id)
        try:
            filepaths = os.listdir(survey_dir)
        except FileNotFoundError:
            continue

        # For each survey
        for fp in filepaths:
            if not fp.endswith(".csv"):
                continue
            try:
                f = pd.read_csv(os.path.join(survey_dir, fp))
            except FileNotFoundError:
                logger.error(f"File not found at path {fp}")
                continue
            except pd.errors.EmptyDataError:
                logger.error(f"No data found at path {fp}")
                continue
            except pd.errors.ParserError:
                logger.error(f"Parse error for file at path {fp}")
                continue
            # Check whether participant uses iOS
            if "event" in f.columns:  # iOS: last columnname == "event"
                # Note: this assumes that all files have headers (check!)
                # Logic for iPhones ###

                # Here you could have a loop over pd.unique(f["survey id"])
                # to do it in
                # one iteration for all surveys -->
                # for sid in survey_ids:
                # Note to Nellie: might be useful to have it iterate over
                # surveys and store timings for each survey

                # select relevant rows and columns
                f = f.loc[(f["survey id"] == survey_id) &  # only this survey
                          ((f["event"] == "present") |  # present/submit event
                           (f["event"] == "submitted")),
                          ["timestamp", "UTC time", "survey id", "event"]]

                # Extract time indicators
                # We assume participants enter only 1 survey per hour
                f["UTC time"] = f["UTC time"].astype("datetime64[ns]")
                f["date_hour"] = f["UTC time"].dt.strftime("%Y_%m_%d_%H")

                # sort by UTC_time
                f = f.sort_values(by="date_hour", ascending=True)

                f = f.drop_duplicates(subset=["date_hour", "event"],
                                      keep="first")

                f = f.pivot(
                    columns="event", values="UTC time", index="date_hour"
                ).reset_index()

                for timestamp in pd.unique(f["date_hour"]):
                    try:
                        present = f.loc[
                            f["date_hour"] == timestamp, "present"
                        ][0]
                    except KeyError:
                        present = None

                    try:
                        submitted = f.loc[
                            f["date_hour"] == timestamp, "submitted"
                        ][0]
                    except KeyError:
                        submitted = None

                    record = np.vstack([record,
                                        [pid, "iOs", timestamp,
                                         present, submitted]])
            else:
                # LOGIC FOR ANDROID USERS
                f = f.loc[(f["survey id"] == survey_id) &  # only this survey
                          ((f["question id"] ==
                            "Survey first rendered and displayed to user") |
                           # only present / submit events
                           (f["question id"] == "User hit submit")),
                          ["timestamp", "UTC time", "question id"]]

                # Extract time indicators
                # We assume participants enter only 1 survey per hour
                f["UTC time"] = f["UTC time"].astype("datetime64[ns]")
                f["date_hour"] = f["UTC time"].dt.strftime("%Y_%m_%d_%H")

                f = f.sort_values(by="date_hour", ascending=True)

                # Looks like if Androids have double events, you should take
                # the   last
                f = f.drop_duplicates(subset=["date_hour", "question id"],
                                      keep="last")

                f = f.pivot(
                    columns="question id", values="UTC time", index="date_hour"
                ).rename(
                    columns={"Survey first rendered and displayed to user":
                             "present",
                             "User hit submit": "submitted"}
                ).reset_index()

                for timestamp in pd.unique(f["date_hour"]):
                    try:
                        present = f["present"][0]
                    except KeyError:
                        present = None

                    try:
                        submitted = f["submitted"][0]
                    except KeyError:
                        submitted = None

                    record = np.vstack([record,
                                        [pid, "Android", timestamp,
                                         present, submitted]])

    svtm = pd.DataFrame(record[1:, :], columns=record[0])

    # Fix surveys that were completed over more than an hour
    svtm["day"] = pd.to_datetime(
        svtm["date_hour"].astype("str"), format="%Y_%m_%d_%H"
    ).dt.strftime("%Y-%m-%d")

    svtm = svtm.groupby(["beiwe_id", "day", "phone_os"]).agg(
        {"start_time": min, "end_time": max}  # for sum durations
    ).reset_index()

    svtm["duration"] = svtm["end_time"] - svtm["start_time"]
    svtm["duration_in_sec"] = svtm["duration"].dt.seconds

    return svtm
