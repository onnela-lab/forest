"""Module for survey submits and survey schedule generation"""
import datetime
import logging
import math
from typing import Optional, Dict

import numpy as np
import pandas as pd

from forest.constants import Frequency
from forest.sycamore.common import read_json
from forest.sycamore.read_audio import get_audio_survey_id_dict

logger = logging.getLogger(__name__)


def convert_time_to_date(submit_time: datetime.datetime,
                         day: int,
                         time_list: list) -> list:
    """Convert an array of times to date

    Takes a single array of timings and a single day

    Args:
        submit_time(datetime):
            date in week for which we want to extract another date and time
        day(int):
            desired day of week
        time_list(list):
            List of timings times from the configuration surveys information

    Returns:
        List of dates corresponding to the times
    """
    # Convert inputted desired day into an integer between 0 and 6
    day %= 7
    # Get the days of the given week using the dow of the given submit day
    dow = submit_time.weekday()
    days = [submit_time + datetime.timedelta(days=i)
            for i in range(0 - dow, 7 - dow)]

    time_list = [str(datetime.timedelta(seconds=time)) for time in time_list]
    time_list = [time.split(":") for time in time_list]
    time_list = [[int(p) for p in time] for time in time_list]

    # Get rid of timing
    #     https://stackoverflow.com/questions/26882499/reset-time-part-of-a-pandas-timestamp
    days = [d - pd.offsets.Micro(0) for d in days]
    list_of_days = [
        [
            d.replace(
                hour=time[0], minute=time[1], second=time[2], microsecond=0
            )
            for time in time_list
        ] for d in days
    ]

    return list_of_days[day]


def generate_survey_times(
        time_start: str, time_end: str, timings: Optional[list] = None,
        survey_type: str = "weekly", intervention_dict: Optional[dict] = None
) -> list:
    """Get delivery times for a survey

    Takes a start time and end time and generates a schedule of all sent
    surveys in time frame for the given survey type

    Args:
        time_start(str):
            The first date for which we want to generate survey times
        time_end(str):
            The last date for which we want to generate survey times
        timings(list):
            list of survey timings, directly from the configuration file
            survey information
        survey_type(str):
            What type of survey schedule to generate times for
        intervention_dict(dict):
            A dictionary with keys for each intervention time, each containing
            a timestamp object
            (only needed for relative surveys)

    Returns:
        A list of all survey times that occur between the time_start and
        time_end per the given survey timings schedule
    """
    if survey_type not in ["weekly", "absolute", "relative"]:
        raise ValueError("Incorrect type of survey."
                         " Ensure this is weekly, absolute, or relative.")

    if timings is None:
        timings = []

        # Get the number of weeks between start and end time
    t_start = pd.Timestamp(time_start)
    t_end = pd.Timestamp(time_end)
    surveys = []

    if survey_type == "weekly":

        weeks = pd.Timedelta(t_end - t_start).days
        # Get ceiling number of weeks
        weeks = math.ceil(weeks / 7.0)

        # Roll dates
        t_lag = list(np.roll(np.array(timings, dtype="object"), -1))

        # for each week, generate the survey times and append to a list
        start_dates = [
            t_start + datetime.timedelta(days=7 * i) for i in range(weeks)
        ]

        for start in start_dates:
            for i, time in enumerate(t_lag):
                if len(time) > 0:
                    surveys.extend(
                        convert_time_to_date(start, day=i, time_list=time)
                    )
    if survey_type == "absolute":
        times_df = pd.DataFrame(timings)
        times_df.columns = ["year", "month", "day", "second"]
        surveys = pd.to_datetime(times_df).tolist()

    if survey_type == "relative":
        if intervention_dict is None:
            logger.error("Error: No dictionary of interventions provided.")
            return []
        for time in timings:
            # This could probably be vectorized
            # time exists for the user
            if intervention_dict[time[0]] is not None:
                # first, get the intervention time.
                # Then, add the days and time of day for the survey to the time
                current_time = (
                    datetime.datetime.fromisoformat(
                        intervention_dict[time[0]]
                    )
                    + pd.Timedelta(time[1], unit="days")
                    + pd.Timedelta(time[2], unit="seconds")
                )  # seconds after time
                surveys.append(current_time)
    return surveys


def get_question_ids(survey_dict: dict, audio_survey_id_dict: dict) -> list:
    """Get question IDs from a survey's dict object.

    Args:
        survey_dict: A dict with information from a specific survey. For
            example, this would come from read_json(config_path)["surveys"][0]
        audio_survey_id_dict: Output from get_audio_survey_id_dict. Dict with
            survey prompts as keys and survey IDs as values

    Returns:
        List of all question IDs for an individual. For audio surveys, it will
        create a "question ID" that is identical to the survey ID

    """
    question_ids = []
    for question in survey_dict["content"]:
        if "question_id" in question.keys():
            question_ids.append(question["question_id"])
        elif "prompt" in question.keys():
            audio_prompt = question["prompt"]
            if audio_prompt not in audio_survey_id_dict.keys():
                logger.warning(
                    "Unable to find survey ID for audio prompt %s",
                    audio_prompt
                )
                continue
            question_ids.append(audio_survey_id_dict[audio_prompt])
    return question_ids


def gen_survey_schedule(
        config_path: str, time_start: str, time_end: str, users: list,
        all_interventions_dict: dict, history_path: Optional[str] = None
) -> pd.DataFrame:
    """Get survey schedule for a number of users

    This generates a DataFrame with the times surveys were delivered for each
    user. In addition, another delivery time is added a week after the last
    delivery time to catch the survey submitted after the last delivery.

    Args:
        config_path(str):
            File path to study configuration file
        time_start(str):
            The first date of the survey data, in YYYY-MM-DD format
        time_end(str):
            The last date of the survey data, in YYYY-MM-DD format
        users(list):
            List of users in study for which we are generating a survey
            schedule
        all_interventions_dict(dict):
            Dictionary containing a key for every beiwe id.
            Each value in the dict is a dict, with a key for each intervention
            and a timestamp for each intervention time
        history_path: Filepath to the survey history file. If this is not
            included, audio survey timings cannot be estimated.

    Returns:
        DataFrame with a line for every survey deployed to every user in
        the study for the given time range
    """
    audio_survey_id_dict = get_audio_survey_id_dict(history_path)
    # List of surveys
    surveys = read_json(config_path)["surveys"]
    # For each survey create a list of survey times
    times_sur = []
    for user in users:
        for i, survey in enumerate(surveys):
            s_times: list = []
            if survey["timings"]:
                s_times = s_times + generate_survey_times(
                    time_start, time_end, timings=survey["timings"],
                    survey_type="weekly"
                )
            if survey["absolute_timings"]:
                s_times = s_times + generate_survey_times(
                    time_start, time_end, timings=survey["absolute_timings"],
                    survey_type="absolute"
                )
            if survey["relative_timings"]:
                # We can only get relative timings if we have an index time
                if user in all_interventions_dict.keys():
                    s_times = s_times + generate_survey_times(
                        time_start, time_end,
                        timings=survey["relative_timings"],
                        survey_type="relative",
                        intervention_dict=all_interventions_dict[user]
                    )
                else:
                    logger.warning("error: no intervention time found for %s",
                                   user)
            tbl = pd.DataFrame({"delivery_time": s_times})
            # May not be necessary, but I"m leaving this in case timestamps are
            # in different formats
            tbl["delivery_time"] = pd.to_datetime(tbl["delivery_time"])
            # add a delivery time a week in the future to capture the last
            # delivery
            week_after_last = tbl.delivery_time.max() + pd.Timedelta(7, "d")
            tbl = pd.concat(
                [tbl, pd.DataFrame({"delivery_time": [week_after_last]})],
                axis=0
            )
            tbl.sort_values("delivery_time", inplace=True)
            tbl.reset_index(drop=True, inplace=True)
            # Create the "next" time column too, which indicates the next time
            # the survey will be deployed
            tbl["next_delivery_time"] = tbl.delivery_time.shift(-1)
            # Remove the placeholder delivery times which were only necessary
            # for calculating the next_delivery_time column
            tbl = tbl.loc[tbl["delivery_time"] != week_after_last, ]
            # remove any rows outside our time interval
            tbl = tbl.loc[(pd.to_datetime(time_start)
                          < tbl["delivery_time"])
                          & (tbl["delivery_time"]
                          < pd.to_datetime(time_end)), ]
            tbl["id"] = i
            tbl["beiwe_id"] = user
            # Get all question IDs for the survey
            question_ids = get_question_ids(survey, audio_survey_id_dict)
            if len(question_ids) > 0:
                q_ids = pd.DataFrame({"question_id": question_ids})
                tbl = pd.merge(tbl, q_ids, how="cross")
            times_sur.append(tbl)
    if len(times_sur) > 0:
        times_sur_df = pd.concat(times_sur).reset_index(drop=True)
    else:
        times_sur_df = pd.DataFrame(
            columns=["delivery_time", "next_delivery_time", "id", "beiwe_id"]
        )
    return times_sur_df


def survey_submits(
        config_path: str, time_start: str, time_end: str, users: list,
        aggregated_data: pd.DataFrame,
        interventions_filepath: Optional[str] = None,
        history_path: Optional[str] = None
) -> pd.DataFrame:
    """Get survey submits for users

    Args:
        config_path(str):
            File path to study configuration file
        time_start(str):
            The first date of the survey data, in YYYY-MM-DD format
        time_end(str):
            The last date of the survey data, in YYYY-MM-DD format
        users(list):
            List of users in study for which we are generating a survey
            schedule
        interventions_filepath(str):
            filepath where interventions json file is.
        aggregated_data(DataFrame):
            Dataframe of aggregated data. Output from aggregate_surveys_config
        history_path: Filepath to the survey history file. If this is not
            included, audio survey timings cannot be estimated.

    Returns:
        A DataFrame with deployment time and information about each user's
        response to a survey for all surveys deployed in the given timeframe.
    """
    agg = aggregated_data.copy()
    if interventions_filepath is None:
        all_interventions_dict = {}
    else:
        all_interventions_dict = get_all_interventions_dict(
            interventions_filepath)

    sched = gen_survey_schedule(config_path, time_start, time_end, users,
                                all_interventions_dict, history_path)
    if sched.shape[0] == 0:  # return empty dataframe
        logger.error("Error: No survey schedules found")
        return pd.DataFrame(columns=[["survey id", "beiwe_id"]])

    # First, figure out if they opened the survey (if there are any lines
    # related to the survey).
    # They get an opened_line if their current surv_inst_flg doesn't match the
    # previous or if the current survey id doesn't match the previous.

    # In some surveys, there will be multiple config_ids for the same line
    # because question IDs are duplicated across surveys. So, all of these
    # config_ids get set with 1 because they all happened at the same time.
    for user in users:
        opening_times = agg.loc[
            (agg.beiwe_id == user) &
            ((agg.surv_inst_flg !=
              agg["surv_inst_flg"].shift(1, fill_value=-1)) |
             (agg["survey id"] !=
             agg["survey id"].shift(1, fill_value="fill"))),
            "Local time"
        ].unique()
        if len(opening_times) == 0:
            continue
        agg.loc[agg.beiwe_id == user, "opened_line"] = agg.loc[
            agg.beiwe_id == user, "Local time"
        ].apply(lambda x: 1 if x in opening_times else 0)
    # Get starting times for each survey to allow joining with files in
    # the by_surveys folder.
    agg["Local time"] = pd.to_datetime(agg["Local time"])
    agg["start_time"] = agg.groupby(
        ["beiwe_id", "survey id", "surv_inst_flg"]
    )["Local time"].transform("first")
    # get rid of answers that were blank responses
    only_real_answers = agg.loc[
        ~agg.answer.isin(["", np.nan, None, [], "[]", "NO_ANSWER_SELECTED"]),
        ["beiwe_id", "survey id", "surv_inst_flg", "question id"]]
    # Look for the number of unique questions where the answer was a real thing
    only_real_answers["num_questions_answered"] = only_real_answers.groupby(
        ["beiwe_id", "survey id", "surv_inst_flg"]
    )["question id"].transform(lambda x: x.nunique())
    # When we do the join, we don't want to create any additional rows in the
    # agg dataframe by having duplicate rows. We also don't want to have the
    # duplicate column.
    only_real_answers = only_real_answers[
        ["beiwe_id", "survey id", "surv_inst_flg", "num_questions_answered"]
    ].drop_duplicates()
    # Append the num_questions_answered column to agg dataframe
    agg = pd.merge(
        agg, only_real_answers,
        # We have to drop 'question id' so that rows with no answer will still
        # get a num_questions_answered
        on=["beiwe_id", "survey id", "surv_inst_flg"],
        how="left"
    )
    # any rows that didn't get joined will be 0 for now
    agg["num_questions_answered"] = agg["num_questions_answered"].fillna(0)

    # Merge survey submit lines onto the schedule data and identify submitted
    # lines
    # This creates a very long dataframe with all survey submissions/openings
    # on lines with each delivery
    submit_lines = pd.merge(
        sched[
            ["delivery_time", "next_delivery_time", "id", "beiwe_id"]
        ].drop_duplicates(),
        agg[
            ["Local time", "config_id", "survey id", "beiwe_id", "submit_line",
             "opened_line", "start_time", "num_questions_answered"]
        ].loc[(agg.submit_line == 1) |
              (agg.opened_line == 1)].drop_duplicates(),
        how="left", left_on=["id", "beiwe_id"],
        right_on=["config_id", "beiwe_id"]
    )
    # Get the submitted survey line
    submit_lines["submit_flg"] = np.where(
        (submit_lines["Local time"] >= submit_lines["delivery_time"]) &
        (submit_lines["Local time"] < submit_lines["next_delivery_time"]) &
        (submit_lines["submit_line"] == 1),
        1, 0
    )
    # find out if they opened a survey within the interval
    submit_lines["opened_flg"] = np.where(
        ((submit_lines["Local time"] >= submit_lines["delivery_time"]) &
         (submit_lines["Local time"] < submit_lines["next_delivery_time"]) &
         (submit_lines["opened_line"] == 1)) |
        # If they submitted the survey, it was open sometime in the interval.
        (submit_lines["submit_flg"] == 1),
        1, 0
    )
    # If they didn't open, they couldn't have started the survey
    submit_lines["start_time"] = np.where(
        (submit_lines["opened_flg"] == 0) & (submit_lines["submit_flg"] == 0),
        np.datetime64("NaT"), submit_lines["start_time"]
    )
    # If they didn't open, it doesn't make sense to have questions answered
    submit_lines["num_questions_answered"] = np.where(
        (submit_lines["opened_flg"] == 0) & (submit_lines["submit_flg"] == 0),
        np.nan, submit_lines["num_questions_answered"]
    )
    # Find whether there were any submissions or openings in each time period
    submit_lines2 = submit_lines.groupby(
        ["delivery_time", "next_delivery_time", "survey id", "beiwe_id",
         "config_id"]
    )[["opened_flg", "submit_flg"]].max().reset_index()
    for col in ["delivery_time", "next_delivery_time"]:
        submit_lines2[col] = pd.to_datetime(submit_lines2[col])
    # Merge on the times of the survey submission
    merge_cols = ["delivery_time", "next_delivery_time", "survey id",
                  "beiwe_id", "config_id", "submit_flg", "opened_flg"]
    submit_lines3 = pd.merge(
        submit_lines2, submit_lines[
            merge_cols + ["Local time", "start_time", "num_questions_answered"]
        ],
        how="left", left_on=merge_cols, right_on=merge_cols
    )
    submit_lines3["submit_time"] = np.where(
        submit_lines3.submit_flg == 1,
        submit_lines3["Local time"],
        np.array(0, dtype="datetime64[ns]")
    )
    # Select appropriate columns
    submit_lines3 = submit_lines3[
        ["survey id", "delivery_time", "beiwe_id", "submit_flg", "opened_flg",
         "submit_time", "start_time", "num_questions_answered"]
    ]
    submit_lines3["time_to_submit"] = (
            submit_lines3["submit_time"] - submit_lines3["delivery_time"]
    ).dt.total_seconds()
    submit_lines3["time_to_open"] = (
            submit_lines3["start_time"] - submit_lines3["delivery_time"]
    ).dt.total_seconds()
    submit_lines3["survey_duration"] = (
            submit_lines3["submit_time"] - submit_lines3["start_time"]
    ).dt.total_seconds()
    # If only a survey_answers file was found, there will be no useful survey
    # duration information because we only have the start time
    submit_lines3["survey_duration"] = np.where(
        submit_lines3["survey_duration"] == 0, np.nan,
        submit_lines3["survey_duration"]
    )
    # If the individual didn't submit the survey, survey duration doesn't make
    # sense (We are only going to define this for complete surveys)
    submit_lines3["survey_duration"] = np.where(
        submit_lines3["submit_flg"] == 0, np.nan,
        submit_lines3["survey_duration"]
    )
    # make cols more interpretable as "no survey submitted"
    submit_lines3["submit_time"] = np.where(
        submit_lines3["submit_flg"] == 1,
        submit_lines3["submit_time"],
        np.array("NaT", dtype="datetime64[ns]")
    )
    # mark as NA if no submit happened
    submit_lines3["time_to_submit"] = np.where(
        submit_lines3["submit_flg"] == 1,
        submit_lines3["time_to_submit"],
        np.nan
    )
    submit_lines3["time_to_open"] = np.where(
        submit_lines3["opened_flg"] == 1,
        submit_lines3["time_to_open"],
        np.nan
    )
    return submit_lines3.sort_values(["survey id", "beiwe_id"]
                                     ).drop_duplicates()


def summarize_submits(submits_df: pd.DataFrame,
                      timeunit: Optional[Frequency] = None,
                      summarize_over_survey: bool = True) -> pd.DataFrame:
    """Summarize a survey submits df

    This generates the number of surveys opened and submitted, as well as the
    average time to open and complete a survey over a period of time

    Args:
        submits_df: output from survey_submits
        timeunit: One of None, Frequency.HOURLY or Frequency.DAILY.
            The unit of time on which to summarize the submits. If this is
            None, submits are summarized over the Beiwe ID and survey ID only.
            If this is "Day" or "Hour", submits are summarized over either the
            day or hour in which they were delivered.
        summarize_over_survey: Whether to summarize over survey. If this
            is True, the output will include a separate row for each survey ID
            within each time interval. If this is False, the output will take
            sums and means over all surveys (i.e. num_surveys_submitted will
            include submissions from more than one survey)

    Returns:
        A DataFrame with a row for each beiwe_id,
        survey ID, and time unit found in submits_df. Has columns:
        num_complete_surveys: Number of surveys completed during the time unit
        num_opened_surveys: Number of surveys opened during the time unit
        avg_time_to_submit: Average time between delivery and submission
        avg_time_to_open: Average time between delivery and opening
        avg_duration: Average time between opening and submission
    """
    # copy dataframe because we will be adding cols to it to process it
    submits = submits_df.copy()
    summary_cols = ["beiwe_id"]
    if summarize_over_survey:
        summary_cols = summary_cols + ["survey id"]
    submits["delivery_time"] = pd.to_datetime(submits["delivery_time"])
    if timeunit == Frequency.HOURLY_AND_DAILY:
        logger.warning("Error: summarize_submits cannot calculate both daily"
                       " and hourly summaries at one time. Running daily "
                       "summaries")
        timeunit = Frequency.DAILY
    if timeunit == Frequency.DAILY:
        submits["delivery_time_floor"] = submits["delivery_time"].dt.floor("D")
    elif timeunit == Frequency.HOURLY:
        submits["delivery_time_floor"] = submits["delivery_time"].dt.floor("H")
    if timeunit is not None:
        # round to the nearest desired unit
        submits["year"] = submits["delivery_time_floor"].dt.year
        submits["month"] = submits["delivery_time_floor"].dt.month
        submits["day"] = submits["delivery_time_floor"].dt.day
        summary_cols = summary_cols + ["year", "month", "day"]
    if timeunit == Frequency.HOURLY:
        summary_cols = summary_cols + ["hour"]
        submits["hour"] = submits["delivery_time_floor"].dt.hour
    num_surveys = submits.groupby(summary_cols)["delivery_time"].nunique()
    num_complete_surveys = submits.groupby(summary_cols)["submit_flg"].sum()
    num_opened_surveys = submits.groupby(summary_cols)["opened_flg"].sum()
    if np.sum(submits.submit_flg == 1) > 0:
        # this will work (and only makes sense) if there is at least one row
        # with a survey submit
        avg_time_to_submit = submits.loc[
            submits.submit_flg == 1
            ].groupby(summary_cols)["time_to_submit"].apply(
            lambda x: np.mean(x)
        )
        avg_time_to_open = submits.loc[
            submits.opened_flg == 1
            ].groupby(summary_cols)["time_to_open"].apply(
            lambda x: np.mean(x)
        )
        avg_duration = submits.loc[
            submits.opened_flg == 1
            ].groupby(summary_cols)["survey_duration"].apply(
            lambda x: np.mean(x)
        )
    else:
        # do the groupby on all rows to avoid getting an error
        avg_time_to_submit = submits.groupby(summary_cols)[
            "time_to_submit"
        ].apply(lambda x: pd.to_datetime("NaT"))
        avg_time_to_open = submits.groupby(summary_cols)[
            "time_to_open"
        ].apply(lambda x: np.mean(x))
        avg_duration = submits.groupby(summary_cols)[
            "survey_duration"
        ].apply(lambda x: pd.to_datetime("NaT"))

    submit_lines_summary = pd.concat(
        [num_surveys, num_complete_surveys, num_opened_surveys,
         avg_time_to_submit, avg_time_to_open, avg_duration],
        axis=1
    ).reset_index()
    submit_lines_summary.columns = summary_cols + [
        "num_surveys", "num_complete_surveys", "num_opened_surveys",
        "avg_time_to_submit", "avg_time_to_open", "avg_duration"
    ]
    return submit_lines_summary


def survey_submits_no_config(input_agg: pd.DataFrame) -> pd.DataFrame:
    """Get survey submits without config file

    Alternative function for getting the survey completions (doesn't have
    expected times of surveys)

    Args:
        input_agg(DataFrame):
            Dataframe of Aggregated Data

    Returns:
        Dataframe with one line per survey submission.

    """
    agg = input_agg.copy()

    def summarize_submission(df):
        temp_dict = {"min_time": df.min(), "max_time": df.max()}
        return pd.Series(temp_dict, index=pd.Index(["min_time", "max_time"]))

    agg = agg.groupby(["survey id", "beiwe_id", "surv_inst_flg"])[
        "Local time"
    ].apply(summarize_submission).reset_index()
    agg = agg.pivot(index=["survey id", "beiwe_id", "surv_inst_flg"],
                    columns="level_3",
                    values="Local time").reset_index()
    agg["time_to_complete"] = agg["max_time"] - agg["min_time"]
    agg["time_to_complete"] = [
        time.seconds for time in agg["time_to_complete"]
    ]

    return agg.sort_values(["beiwe_id", "survey id"])


def get_all_interventions_dict(filepath: Optional[str]) -> dict:
    """Read json file into interventions dict

    Extracts user intervention information for use in survey_timings.

    Args:
        filepath: the path to a json file containing patient interventions
            information (downloaded from the beiwe website)

    Returns:
        a dict with one key for each beiwe_id in the study. The value for each
        key is a dict with a key for each intervention in the study, and a
        value which is the timestamp
    """
    if filepath is None:
        return {}
    full_dict = read_json(filepath)
    output_dict: Dict = {}

    for user in full_dict.keys():
        output_dict[user] = {}
        for survey in full_dict[user].keys():
            for time in full_dict[user][survey].keys():
                output_dict[user][time] = full_dict[user][survey][time]

    return output_dict
