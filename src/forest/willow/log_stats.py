"""This module contains functions for calculating summary statistics for the
communication logs.
"""
import logging
import os
from typing import List, Optional

import pandas as pd
import numpy as np

from forest.constants import Frequency
from forest.poplar.legacy.common_funcs import (
    read_data,
    write_all_summaries,
    datetime2stamp,
    stamp2datetime,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_mean_responsiveness(
    df: pd.DataFrame,
    col_with_sent_received: str,
    received_values_list: list,
    sent_values_list: list,
) -> float:
    """Calculate the mean time in minutes between recieving and sending a
    message

    Args:
        df: The dataframe with calls or texts
        col_with_sent_received: The column indicating whether a message is sent
            or received
        received_values_list: values of col_with_sent_received indicating that
            a message/call was received
        sent_values_list: values of col_with_sent_received indicating that a
            message/call was sent

    Returns:
        float: The average time between having a call sent or recieved

    """
    cols = ["hashed phone number", "timestamp"]
    received = df.loc[
        df[col_with_sent_received].isin(received_values_list), cols
    ]

    sent = df.loc[df[col_with_sent_received].isin(sent_values_list), cols]

    # Joining the dataframe to itself will filter to only messages that are
    # received

    joined = pd.merge(
        left=sent,
        right=received,
        how="inner",
        on="hashed phone number",
        suffixes=("_sent", "_received"),
    )
    if joined.shape[0] == 0:
        return pd.NA

    # We only care about instances where a message was sent after being
    # recieved
    joined = joined.loc[
        joined["timestamp_sent"] > joined["timestamp_received"]
    ]

    # Take only the first sent message after each incoming message
    joined.sort_values(
        ["hashed phone number", "timestamp_received", "timestamp_sent"],
        inplace=True,
        ascending=True,
    )
    joined.reset_index(drop=True, inplace=True)
    # We only want to take the first row by received timestamp because that
    # will be the earliest possible sent time
    joined.drop_duplicates(
        subset=["hashed phone number", "timestamp_received"],
        keep="first",
        inplace=True,
    )
    # We now want to ensure that no sent timestamps are tied to two received
    # timestamps
    joined.drop_duplicates(
        subset=["hashed phone number", "timestamp_sent"],
        keep="first",
        inplace=True,
    )

    mean_responsiveness_miliseconds = (
        joined["timestamp_sent"] - joined["timestamp_received"]
    ).mean()

    mean_responsiveness_minutes = mean_responsiveness_miliseconds / 1_000 / 60

    return mean_responsiveness_minutes


def text_analysis(
    df_text: pd.DataFrame, stamp: int, step_size: int, frequency: Frequency
) -> dict:
    """Calculate the summary statistics for the text data
    in the given time interval.

    Args:
        df_text: pd.DataFrame
            dataframe of the text data
        stamp: int
            starting timestamp of the study
        step_size: int
            ending timestamp of the study
        frequency: Frequency class,
            determining resolution of the summary stats

    Returns:
        dict of summary statistics containing:
            num_s: int
                number of sent SMS
            num_r: int
                number of received SMS
            num_mms_s: int
                number of sent MMS
            num_mms_r: int
                number of received MMS
            num_s_tel: int
                number of unique phone numbers in sent SMS
            num_r_tel: int
                number of unique phone numbers in received SMS
            total_char_s: int
                total number of characters in sent SMS
            total_char_r: int
                total number of characters in received SMS
            mean_responsiveness_text: float
                Mean number of minutes between a received text and a sent text
            text_reciprocity_incoming: int
                number of received SMS without response
            text_reciprocity_outgoing: int
                number of sent SMS without response

    """
    # filter the data based on the timestamp
    temp_text = df_text[
        (df_text["timestamp"] / 1000 >= stamp)
        & (df_text["timestamp"] / 1000 < stamp + step_size)
    ]

    mean_responsiveness_text = get_mean_responsiveness(
        df=temp_text,
        col_with_sent_received="sent vs received",
        received_values_list=["received SMS", "received MMS"],
        sent_values_list=["sent SMS", "sent MMS"],
    )

    # calculate the number of texts
    message_lengths = np.array(temp_text["message length"])
    for k, mlength in enumerate(message_lengths):
        if mlength == "MMS":
            message_lengths[k] = 0
        if not isinstance(mlength, str):
            if np.isnan(mlength):
                message_lengths[k] = 0

    message_lengths = message_lengths.astype(int)

    index_s = np.array(temp_text["sent vs received"]) == "sent SMS"
    index_r = np.array(temp_text["sent vs received"]) == "received SMS"

    send_to_number = np.unique(
        np.array(temp_text["hashed phone number"])[index_s]
    )
    receive_from_number = np.unique(
        np.array(temp_text["hashed phone number"])[index_r]
    )

    num_s_tel = len(send_to_number)
    num_r_tel = len(receive_from_number)

    index_mms_s = np.array(temp_text["sent vs received"]) == "sent MMS"
    index_mms_r = np.array(temp_text["sent vs received"]) == "received MMS"

    num_s = sum(index_s.astype(int))
    num_r = sum(index_r.astype(int))
    num_mms_s = sum(index_mms_s.astype(int))
    num_mms_r = sum(index_mms_r.astype(int))
    total_char_s = sum(message_lengths[index_s])
    total_char_r = sum(message_lengths[index_r])

    text_reciprocity_incoming = None
    text_reciprocity_outgoing = None

    if frequency == Frequency.DAILY:
        # find the phone number in sent_from, but not in send_to
        received_no_response = [
            tel for tel in receive_from_number if tel not in send_to_number
        ]
        sent_no_response = [
            tel for tel in send_to_number if tel not in receive_from_number
        ]

        text_reciprocity_incoming = 0
        for tel in received_no_response:
            text_reciprocity_incoming += sum(
                index_r * (np.array(temp_text["hashed phone number"]) == tel)
            )

        text_reciprocity_outgoing = 0
        for tel in sent_no_response:
            text_reciprocity_outgoing += sum(
                index_s * (np.array(temp_text["hashed phone number"]) == tel)
            )

    return {
        "num_s": num_s,
        "num_r": num_r,
        "num_mms_s": num_mms_s,
        "num_mms_r": num_mms_r,
        "num_s_tel": num_s_tel,
        "num_r_tel": num_r_tel,
        "total_char_s": total_char_s,
        "total_char_r": total_char_r,
        "mean_responsiveness_text": mean_responsiveness_text,
        "text_reciprocity_incoming": text_reciprocity_incoming,
        "text_reciprocity_outgoing": text_reciprocity_outgoing,
    }


def text_and_call_analysis(
    df_call: pd.DataFrame, df_text: pd.DataFrame, stamp: int, step_size: int
) -> dict:
    """Calculate the summary statistics for anything requiring both call and
    text data in the given time interval.
    Args:
        df_call: pd.DataFrame
            dataframe of the call data
        df_text: pd.DataFrame
            dataframe of the text data
        stamp: int
            starting timestamp of the interval
        step_size: int
            ending timestamp of the interval

    Returns:
        dict of summary statistics containing:
            num_uniq_individuals_call_or_text: int
                number of people making incoming calls or texts to the Beiwe
                user or who the Beiwe user made outgoing calls or texts to


    """
    # filter the data based on the timestamp
    if df_call.shape[0] > 0:
        temp_call = df_call[
            (df_call["timestamp"] / 1000 >= stamp)
            & (df_call["timestamp"] / 1000 < stamp + step_size)
        ]
        index_in_call = np.array(temp_call["call type"]) == "Incoming Call"
        index_out_call = np.array(temp_call["call type"]) == "Outgoing Call"
        index_mis_call = np.array(temp_call["call type"]) == "Missed Call"
        calls_in = np.array(temp_call["hashed phone number"])[index_in_call]
        calls_out = np.array(temp_call["hashed phone number"])[index_out_call]
        calls_mis = np.array(temp_call["hashed phone number"])[index_mis_call]

    else:  # no calls were received, so no unique numbers will be used
        calls_in = np.array([])
        calls_out = np.array([])
        calls_mis = np.array([])

    if df_text.shape[0] > 0:
        temp_text = df_text[
            (df_text["timestamp"] / 1000 >= stamp)
            & (df_text["timestamp"] / 1000 < stamp + step_size)
        ]

        index_s = np.array(temp_text["sent vs received"]) == "sent SMS"
        index_r = np.array(temp_text["sent vs received"]) == "received SMS"
        texts_in = np.array(temp_text["hashed phone number"])[index_r]
        texts_out = np.array(temp_text["hashed phone number"])[index_s]
    else:  # no texts were received, so no unique numbers will be used
        texts_in = np.array([])
        texts_out = np.array([])

    num_uniq_individuals_call_or_text = len(
        np.unique(
            np.hstack([calls_in, texts_in, texts_out, calls_out, calls_mis])
        )
    )
    return {
        "num_uniq_individuals_call_or_text": num_uniq_individuals_call_or_text,
    }


def get_call_reciprocity(calls_dict: dict) -> float:
    """
    Get call reciprocity for an individual.
    This is defined as 1 - (|incoming - outgoing|) / (incoming + outgoing).
    A reciprocity of 1 indicates perfect reciprocity--all incoming calls are
    balanced by an outgoing call. A reciprocity of 0 indicates that an
    individual had only incoming calls OR that they had only outgoing calls.

    Args:
       calls_dict: a dict with two keys: "incoming" and "outgoing".
           "incoming" includes a series with incoming phone numbers, and
           "outgoing" has outgoing phone numbers

    Returns: Reciprocity index

    """

    value_counts = dict()

    for k in ["incoming", "outgoing"]:
        value_counts[k] = (
            pd.Series(calls_dict[k])
            .value_counts()
            .reset_index()
            .rename(
                {"index": "hashed_phone_number", "count": "num_" + k}, axis=1
            )
        )

    merged_value_counts = pd.merge(
        left=value_counts["incoming"],
        right=value_counts["outgoing"],
        how="outer",
        on="hashed_phone_number",
    ).fillna(0)

    if merged_value_counts.empty:
        return pd.NA

    merged_value_counts["weight"] = (
        merged_value_counts["num_incoming"]
        + merged_value_counts["num_outgoing"]
    )

    merged_value_counts["reciprocity"] = 1 - (
        np.abs(
            merged_value_counts["num_incoming"]
            - merged_value_counts["num_outgoing"]
        )
        / (
            merged_value_counts["num_incoming"]
            + merged_value_counts["num_outgoing"]
        )
    )

    return (
        merged_value_counts["reciprocity"] * merged_value_counts["weight"]
    ).sum() / merged_value_counts["weight"].sum()


def call_analysis(df_call: pd.DataFrame, stamp: int, step_size: int) -> dict:
    """Calculate the summary statistics for the call data
    in the given time interval.

    Args:
        df_call: pd.DataFrame
            dataframe of the call data
        stamp: int
            starting timestamp of the interval
        step_size: int
            ending timestamp of the interval

    Returns:
        tuple of summary statistics containing:
            num_in_call: int
                number of incoming calls
            num_out_call: int
                number of outgoing calls
            num_mis_call: int
                number of missed calls
            num_uniq_in_call: int
                number of unique phone numbers in incoming calls
            num_uniq_out_call: int
                number of unique phone numbers in outgoing calls
            num_uniq_mis_call: int
                number of unique phone numbers in missed calls
            total_time_in_call: int
                total time in minutes of incoming calls
            total_time_out_call: int
                total time in minutes of outgoing calls
            mean_responsiveness_call: float
                Mean number of minutes between a received call and a sent call
            call_reciprocity: float
                Reciprocity of calls.
    """
    # filter the data based on the timestamp
    temp_call = df_call[
        (df_call["timestamp"] / 1000 >= stamp)
        & (df_call["timestamp"] / 1000 < stamp + step_size)
    ]

    mean_resposiveness_call = get_mean_responsiveness(
        df=temp_call,
        col_with_sent_received="call type",
        received_values_list=["Incoming Call", "Missed Call"],
        sent_values_list=["Outgoing Call"],
    )

    dur_in_sec = np.array(temp_call["duration in seconds"])
    dur_in_sec[np.isnan(dur_in_sec)] = 0
    dur_in_min = dur_in_sec / 60

    index_in_call = np.array(temp_call["call type"]) == "Incoming Call"
    index_out_call = np.array(temp_call["call type"]) == "Outgoing Call"
    index_mis_call = np.array(temp_call["call type"]) == "Missed Call"

    num_in_call = sum(index_in_call)
    num_out_call = sum(index_out_call)
    num_mis_call = sum(index_mis_call)

    num_uniq_in_call = len(
        np.unique(np.array(temp_call["hashed phone number"])[index_in_call])
    )
    num_uniq_out_call = len(
        np.unique(np.array(temp_call["hashed phone number"])[index_out_call])
    )
    num_uniq_mis_call = len(
        np.unique(np.array(temp_call["hashed phone number"])[index_mis_call])
    )

    total_time_in_call = sum(dur_in_min[index_in_call])
    total_time_out_call = sum(dur_in_min[index_out_call])

    call_reciprocity = get_call_reciprocity(
        {
            "incoming": np.array(temp_call["hashed phone number"])[
                index_in_call | index_mis_call
            ],
            "outgoing": np.array(temp_call["hashed phone number"])[
                index_out_call
            ],
        }
    )

    return {
        "num_in_call": num_in_call,
        "num_out_call": num_out_call,
        "num_mis_call": num_mis_call,
        "num_in_caller": num_uniq_in_call,
        "num_out_caller": num_uniq_out_call,
        "num_mis_caller": num_uniq_mis_call,
        "total_mins_in_call": total_time_in_call,
        "total_mins_out_call": total_time_out_call,
        "mean_resposiveness_call": mean_resposiveness_call,
        "call_reciprocity": call_reciprocity,
    }


def comm_logs_summaries(
    df_text: pd.DataFrame,
    df_call: pd.DataFrame,
    stamp_start: float,
    stamp_end: float,
    tz_str: str,
    frequency: Frequency,
) -> pd.DataFrame:
    """Calculate the summary statistics for the communication logs.

    Args:
        df_text: pd.DataFrame
            dataframe of the text data
        df_call: pd.DataFrame
            dataframe of the call data
        stamp_start: int
            starting timestamp of the interval
        stamp_end: int
            ending timestamp of the interval
        tz_str: str
            timezone where the study was/is conducted
        frequency: Frequency class,
            determining resolution of the summary stats

    Returns:
        pandas dataframe of summary stats
    """
    all_summary_stats = []
    start_year, start_month, start_day, start_hour, _, _ = stamp2datetime(
        stamp_start, tz_str
    )
    end_year, end_month, end_day, end_hour, _, _ = stamp2datetime(
        stamp_end, tz_str
    )

    # determine the starting and ending timestamp again based on the frequency
    if frequency == Frequency.HOURLY_AND_DAILY:
        logger.error(
            "Error: frequency cannot be HOURLY_AND_DAILY for this function"
        )

    if frequency == Frequency.DAILY:
        table_start = datetime2stamp(
            (start_year, start_month, start_day, 0, 0, 0), tz_str
        )
        table_end = datetime2stamp(
            (end_year, end_month, end_day, 0, 0, 0), tz_str
        )
    else:
        table_start = datetime2stamp(
            [start_year, start_month, start_day, start_hour, 0, 0], tz_str
        )
        table_end = datetime2stamp(
            [end_year, end_month, end_day, end_hour, 0, 0], tz_str
        )

    # determine the step size based on the frequency
    # step_size is in seconds
    step_size = 60 * frequency.value

    # for each chunk, calculate the summary statistics (colmean or count)
    for stamp in np.arange(table_start, table_end + 1, step=step_size):
        year, month, day, hour, _, _ = stamp2datetime(int(stamp), tz_str)
        # initialize the summary statistics
        current_line = dict()

        if df_call.shape[0] > 0:
            call_stats = call_analysis(df_call, int(stamp), step_size)
            current_line.update(call_stats)
        if df_text.shape[0] > 0 or df_call.shape[0] > 0:
            text_and_call_stats = text_and_call_analysis(
                df_call, df_text, int(stamp), step_size
            )
            current_line.update(text_and_call_stats)

        if df_text.shape[0] > 0:
            text_stats = text_analysis(df_text, int(stamp), step_size, frequency)
            current_line.update(text_stats)

        if frequency == Frequency.DAILY:
            current_line.update({"year": year, "month": month, "day": day})
        else:
            current_line.update(
                {"year": year, "month": month, "day": day, "hour": hour}
            )

        all_summary_stats.append(current_line)

    call_columns = [
        "num_in_call",
        "num_out_call",
        "num_mis_call",
        "num_in_caller",
        "num_out_caller",
        "num_mis_caller",
        "total_mins_in_call",
        "total_mins_out_call",
    ]
    call_columns_daily_only = ["mean_resposiveness_call", "call_reciprocity"]

    call_and_text_columns = ["num_uniq_individuals_call_or_text"]

    text_columns = [
        "num_s",
        "num_r",
        "num_mms_s",
        "num_mms_r",
        "num_s_tel",
        "num_r_tel",
        "total_char_s",
        "total_char_r",
    ]

    text_columns_daily_only = [
        "mean_responsiveness_text",
        "text_reciprocity_incoming",
        "text_reciprocity_outgoing",
    ]

    if frequency == Frequency.DAILY:
        columns_to_output = (
            ["year", "month", "day"]
            + call_columns
            + call_columns_daily_only
            + call_and_text_columns
            + text_columns
            + text_columns_daily_only
        )
    elif frequency == Frequency.HOURLY:
        columns_to_output = (
            ["year", "month", "day", "hour"]
            + call_columns
            + call_and_text_columns
            + text_columns
        )
    else:
        raise NotImplementedError(
            "Willow only supports hourly and daily aggregation"
        )

    output_lines = []

    for line_dict in all_summary_stats:
        current_line_dict = [
            line_dict.get(col, pd.NA) for col in columns_to_output
        ]
        output_lines.append(current_line_dict)

    return pd.DataFrame(output_lines, columns=columns_to_output)


# Main function/wrapper should take standard arguments with Beiwe names:
def log_stats_main(
    study_folder: str,
    output_folder: str,
    tz_str: str,
    frequency: Frequency,
    time_start: Optional[List] = None,
    time_end: Optional[List] = None,
    beiwe_ids: Optional[List[str]] = None,
) -> None:
    """Main function for calculating the summary statistics for the
    communication logs.

    Args:
        study_folder: path to the study folder
        output_folder: path to the output folder
        tz_str: timezone where the study was/is conducted
        frequency: Frequency class,
            determining resolution of the summary stats
        time_start: starting timestamp of the study
        time_end: ending timestamp of the study
        beiwe_ids: list of Beiwe IDs to be processed
    """

    if frequency not in [
        Frequency.HOURLY_AND_DAILY,
        Frequency.DAILY,
        Frequency.HOURLY,
    ]:
        logger.error(
            "Error: frequency must be one of the following: "
            "HOURLY_AND_DAILY, DAILY, HOURLY"
        )

    if frequency == Frequency.HOURLY_AND_DAILY:
        frequencies = [Frequency.HOURLY, Frequency.DAILY]
    else:
        frequencies = [frequency]

    os.makedirs(output_folder, exist_ok=True)
    for freq in frequencies:
        os.makedirs(f"{output_folder}/{freq.name.lower()}", exist_ok=True)

    # beiwe_id should be a list of str
    if beiwe_ids is None:
        beiwe_ids = [
            participant_id
            for participant_id in os.listdir(study_folder)
            if os.path.isdir(f"{study_folder}/{participant_id}")
        ]

    # process the data for each participant in each frequency into a folder of
    # the corresponding frequency.
    for beiwe_id in beiwe_ids:
        for freq in frequencies:
            logger.info("(%s) Participant: %s", freq.name.lower(), beiwe_id)
            try:
                log_stats_inner(
                    beiwe_id,
                    f"{output_folder}/{freq.name.lower()}",
                    study_folder,
                    frequency,
                    tz_str,
                    time_start,
                    time_end,
                )
            except Exception as err:
                logger.error(
                    "An error occurred when processing data: %s", err
                )

    logger.info("Summary statistics obtained. Finished.")


def log_stats_inner(
    beiwe_id: str,
    output_folder: str,
    study_folder: str,
    frequency: Frequency,
    tz_str: str,
    time_start: Optional[List] = None,
    time_end: Optional[List] = None,
):
    """Inner functionality of log_stats_main"""
    # read data
    text_data, text_stamp_start, text_stamp_end = read_data(
        beiwe_id, study_folder, "texts", tz_str, time_start, time_end
    )
    call_data, call_stamp_start, call_stamp_end = read_data(
        beiwe_id, study_folder, "calls", tz_str, time_start, time_end
    )

    # give up early if there is no data
    if text_data.shape[0] <= 0 and call_data.shape[0] <= 0:
        logger.info("There was no data for participant %s", beiwe_id)
        return

    # stamps from call and text should be the stamp_end
    logger.info("Data imported ...")
    stamp_start = min(text_stamp_start, call_stamp_start)
    stamp_end = max(text_stamp_end, call_stamp_end)

    # process the data
    stats_pdframe = comm_logs_summaries(
        text_data, call_data, stamp_start, stamp_end, tz_str, frequency
    )

    # num_uniq_individuals_call_or_text is the cardinality of the union of
    # several sets. It should should always be at least as large as the
    # cardinality of any one of the sets, and it should never be larger than
    # the sum of the cardinalities of all of the sets. (it may be equal if all
    # the sets are disjoint)
    num_uniq_column = "num_uniq_individuals_call_or_text"  # legibility hax.
    sum_all_set_cols = pd.Series([0] * stats_pdframe.shape[0])
    for column in [
        "num_s_tel",
        "num_r_tel",
        "num_in_caller",
        "num_out_caller",
        "num_mis_caller",
    ]:
        sum_all_set_cols += pd.to_numeric(stats_pdframe[column]).fillna(0)
        if (
            pd.to_numeric(stats_pdframe[num_uniq_column]).fillna(0)
            < pd.to_numeric(stats_pdframe[column]).fillna(0)
        ).any():
            logger.error(
                "Error: "
                "num_uniq_individuals_call_or_text was found to be less than "
                "%s for at least one time interval. This error comes from an "
                "issue with the code, not an issue with the input data.",
                column,
            )

    if (
        pd.to_numeric(stats_pdframe[num_uniq_column]).fillna(0)
        > sum_all_set_cols
    ).any():
        logger.error(
            "Error: "
            "num_uniq_individuals_call_or_text was found to be larger than the"
            "sum of individual cardinalities for at least one time interval. "
            "This error comes from an issue with the code, not an issue with "
            "the input data."
        )

    write_all_summaries(beiwe_id, stats_pdframe, output_folder)
