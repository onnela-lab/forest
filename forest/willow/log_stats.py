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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def text_analysis(
    df_text: pd.DataFrame, stamp: int, step_size: int, frequency: Frequency
) -> tuple:
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
        tuple of summary statistics containing:
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
                index_r
                * (np.array(temp_text["hashed phone number"]) == tel)
            )

        text_reciprocity_outgoing = 0
        for tel in sent_no_response:
            text_reciprocity_outgoing += sum(
                index_s
                * (np.array(temp_text["hashed phone number"]) == tel)
            )

    return (
        num_s,
        num_r,
        num_mms_s,
        num_mms_r,
        num_s_tel,
        num_r_tel,
        total_char_s,
        total_char_r,
        text_reciprocity_incoming,
        text_reciprocity_outgoing,
    )


def text_and_call_analysis(
    df_call: pd.DataFrame, df_text: pd.DataFrame, stamp: int, step_size: int
) -> tuple:
    """Calculate the summary statistics for the call and text data
    in the given time interval.

    Args:
        df_call: pd.DataFrame
            dataframe of the call data
        df_text: pd.DataFrame
            dataframe of the text data
        stamp: int
            starting timestamp of the study
        step_size: int
            ending timestamp of the study

    Returns:
        tuple of summary statistics containing:
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

    num_uniq_individuals_call_or_text = len(np.unique(np.hstack(
        [calls_in, texts_in, texts_out, calls_out, calls_mis]
    )))
    return (
        num_uniq_individuals_call_or_text,
    )


def call_analysis(df_call: pd.DataFrame, stamp: int, step_size: int) -> tuple:
    """Calculate the summary statistics for the call data
    in the given time interval.

    Args:
        df_call: pd.DataFrame
            dataframe of the call data
        stamp: int
            starting timestamp of the study
        step_size: int
            ending timestamp of the study

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
    """
    # filter the data based on the timestamp
    temp_call = df_call[
        (df_call["timestamp"] / 1000 >= stamp)
        & (df_call["timestamp"] / 1000 < stamp + step_size)
    ]

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
        np.unique(
            np.array(temp_call["hashed phone number"])[index_in_call]
        )
    )
    num_uniq_out_call = len(
        np.unique(
            np.array(temp_call["hashed phone number"])[index_out_call]
        )
    )
    num_uniq_mis_call = len(
        np.unique(
            np.array(temp_call["hashed phone number"])[index_mis_call]
        )
    )

    total_time_in_call = sum(dur_in_min[index_in_call])
    total_time_out_call = sum(dur_in_min[index_out_call])

    return (
        num_in_call,
        num_out_call,
        num_mis_call,
        num_uniq_in_call,
        num_uniq_out_call,
        num_uniq_mis_call,
        total_time_in_call,
        total_time_out_call,
    )


def comm_logs_summaries(
    df_text: pd.DataFrame, df_call: pd.DataFrame, stamp_start: float,
    stamp_end: float, tz_str: str, frequency: Frequency
) -> pd.DataFrame:
    """Calculate the summary statistics for the communication logs.

    Args:
        df_text: pd.DataFrame
            dataframe of the text data
        df_call: pd.DataFrame
            dataframe of the call data
        stamp_start: int
            starting timestamp of the study
        stamp_end: int
            ending timestamp of the study
        tz_str: str
            timezone where the study was/is conducted
        frequency: Frequency class,
            determining resolution of the summary stats

    Returns:
        pandas dataframe of summary stats

    Raises:
        ValueError: if frequency is not of correct type
    """
    summary_stats = []
    start_year, start_month, start_day, start_hour, _, _ = stamp2datetime(
        stamp_start, tz_str
    )
    end_year, end_month, end_day, end_hour, _, _ = stamp2datetime(
        stamp_end, tz_str
    )

    # determine the starting and ending timestamp again based on the frequency
    if frequency == Frequency.HOURLY_AND_DAILY:
        raise ValueError("frequency not of correct type")

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
    step_size = 3600 * frequency.value

    # for each chunk, calculate the summary statistics (colmean or count)
    for stamp in np.arange(table_start, table_end + 1, step=step_size):
        year, month, day, hour, _, _ = stamp2datetime(stamp, tz_str)
        # initialize the summary statistics
        newline = []

        if df_call.shape[0] > 0:
            call_stats = call_analysis(df_call, stamp, step_size)
            newline += list(call_stats)
        else:
            newline += [pd.NA] * 8
        if df_text.shape[0] > 0 or df_call.shape[0] > 0:
            text_and_call_stats = text_and_call_analysis(
                df_call, df_text, stamp, step_size
            )
            newline += list(text_and_call_stats)
        else:
            newline += [pd.NA]

        if df_text.shape[0] > 0:
            text_stats = text_analysis(df_text, stamp, step_size, frequency)
            newline += list(text_stats)
        else:
            newline += [pd.NA] * 10
        if frequency == Frequency.DAILY:
            newline = [year, month, day] + newline
        else:
            newline = [year, month, day, hour] + newline[:16]

        summary_stats.append(newline)

    columns = [
        "num_in_call",
        "num_out_call",
        "num_mis_call",
        "num_in_caller",
        "num_out_caller",
        "num_mis_caller",
        "total_mins_in_call",
        "total_mins_out_call",
        "num_uniq_individuals_call_or_text",
        "num_s",
        "num_r",
        "num_mms_s",
        "num_mms_r",
        "num_s_tel",
        "num_r_tel",
        "total_char_s",
        "total_char_r",
    ]
    if frequency == Frequency.DAILY:
        return pd.DataFrame(
            summary_stats,
            columns=["year", "month", "day"]
            + columns
            + [
                "text_reciprocity_incoming",
                "text_reciprocity_outgoing",
            ],
        )

    return pd.DataFrame(
        summary_stats,
        columns=["year", "month", "day", "hour"] + columns,
    )


# Main function/wrapper should take standard arguments with Beiwe names:
def log_stats_main(
    study_folder: str,
    output_folder: str,
    tz_str: str,
    frequency: Frequency,
    time_start: Optional[List] = None,
    time_end: Optional[List] = None,
    beiwe_id: Optional[List[str]] = None,
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
        beiwe_id: list of Beiwe IDs to be processed
    """
    os.makedirs(output_folder, exist_ok=True)

    if frequency == Frequency.HOURLY_AND_DAILY:
        os.makedirs(output_folder + "/hourly", exist_ok=True)
        os.makedirs(output_folder + "/daily", exist_ok=True)

    # beiwe_id should be a list of str
    if beiwe_id is None:
        beiwe_id = [
            i for i in os.listdir(study_folder)
            if os.path.isdir(f"{study_folder}/{i}")
        ]

    if len(beiwe_id) > 0:
        for bid in beiwe_id:
            logger.info("User: %s", bid)
            try:
                # read data
                text_data, text_stamp_start, text_stamp_end = read_data(
                    bid, study_folder, "texts", tz_str, time_start, time_end
                )
                call_data, call_stamp_start, call_stamp_end = read_data(
                    bid, study_folder, "calls", tz_str, time_start, time_end
                )

                if text_data.shape[0] > 0 or call_data.shape[0] > 0:
                    # stamps from call and text should be the stamp_end
                    logger.info("Data imported ...")
                    stamp_start = min(text_stamp_start, call_stamp_start)
                    stamp_end = max(text_stamp_end, call_stamp_end)

                    # process data
                    if frequency == Frequency.HOURLY_AND_DAILY:
                        stats_pdframe1 = comm_logs_summaries(
                            text_data,
                            call_data,
                            stamp_start,
                            stamp_end,
                            tz_str,
                            Frequency.HOURLY,
                        )
                        stats_pdframe2 = comm_logs_summaries(
                            text_data,
                            call_data,
                            stamp_start,
                            stamp_end,
                            tz_str,
                            Frequency.DAILY,
                        )

                        write_all_summaries(
                            bid, stats_pdframe1, output_folder + "/hourly"
                        )
                        write_all_summaries(
                            bid, stats_pdframe2, output_folder + "/daily"
                        )
                    else:
                        stats_pdframe = comm_logs_summaries(
                            text_data,
                            call_data,
                            stamp_start,
                            stamp_end,
                            tz_str,
                            frequency,
                        )
                        # num_uniq_individuals_call_or_text is the cardinality
                        # of the union of several sets. It should should always
                        # be at least as large as the cardinality of any one of
                        # the sets, and it should never be larger than the sum
                        # of the cardinalities of all of the sets
                        # (it may be equal if all the sets are disjoint)
                        sum_all_set_cols = pd.Series(
                            [0]*stats_pdframe.shape[0]
                        )
                        for col in [
                            "num_s_tel", "num_r_tel", "num_in_caller",
                            "num_out_caller", "num_mis_caller"
                        ]:
                            sum_all_set_cols += stats_pdframe[col]
                            if (
                                stats_pdframe[
                                    "num_uniq_individuals_call_or_text"
                                ] < stats_pdframe[col]
                            ).any():
                                logger.error(
                                    "Error: "
                                    "num_uniq_individuals_call_or_text "
                                    "was found to be less than %s for at "
                                    "least one time interval. This error "
                                    "comes from an issue with the code,"
                                    " not an issue with the input data",
                                    col
                                    )
                        if (
                            stats_pdframe[
                                "num_uniq_individuals_call_or_text"
                            ] > sum_all_set_cols
                        ).any():
                            logger.error(
                                    "Error: "
                                    "num_uniq_individuals_call_or_text "
                                    "was found to be larger than the sum "
                                    "of individual cardinalities for at "
                                    "least one time interval. This error "
                                    "comes from an issue with the code,"
                                    " not an issue with the input data"
                                    )

                        write_all_summaries(bid, stats_pdframe, output_folder)

                    logger.info(
                        "Summary statistics obtained. Finished."
                    )

            except Exception as err:
                logger.error(
                    "An error occurred when processing the data: %s", err
                )
