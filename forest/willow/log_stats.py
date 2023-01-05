import os
import sys

import pandas as pd
import numpy as np

from forest.constants import Frequency
from forest.poplar.legacy.common_funcs import (
    read_data,
    write_all_summaries,
    datetime2stamp,
    stamp2datetime,
)


def comm_logs_summaries(
    ID: str, df_text, df_call, stamp_start, stamp_end, tz_str, frequency
):
    """
    Docstring
    Args:
        Beiwe ID is needed here only for debugging.
          The other inputs are the outputs from read_comm_logs().
        frequency: Frequency class, determining resolution of the summary stats
        tz_str: timezone where the study was/is conducted
    Return: pandas dataframe of summary stats
    """
    summary_stats = []
    [
        start_year,
        start_month,
        start_day,
        start_hour,
        start_min,
        start_sec,
    ] = stamp2datetime(stamp_start, tz_str)
    [
        end_year, end_month, end_day, end_hour, end_min, end_sec
    ] = stamp2datetime(stamp_end, tz_str)

    # determine the starting and ending timestamp again based on the frequency
    if frequency in [
        Frequency.HOURLY, Frequency.THREE_HOURLY,
        Frequency.SIX_HOURLY, Frequency.TWELVE_HOURLY
    ]:
        table_start = datetime2stamp(
            [start_year, start_month, start_day, start_hour, 0, 0], tz_str
        )
        table_end = datetime2stamp(
            [end_year, end_month, end_day, end_hour, 0, 0], tz_str
        )
        step_size = 3600 * frequency.value
    elif frequency == Frequency.DAILY:
        table_start = datetime2stamp(
            (start_year, start_month, start_day, 0, 0, 0), tz_str
        )
        table_end = datetime2stamp(
            (end_year, end_month, end_day, 0, 0, 0), tz_str
        )
        step_size = 3600 * 24
    else:
        raise ValueError("frequency not of correct type")

    # for each chunk, calculate the summary statistics (colmean or count)
    for stamp in np.arange(table_start, table_end + 1, step=step_size):
        (year, month, day, hour, minute, second) = stamp2datetime(
            stamp, tz_str
        )
        (
            num_in_call,
            num_out_call,
            num_mis_call,
            num_uniq_in_call,
            num_uniq_out_call,
            num_uniq_mis_call,
            total_time_in_call,
            total_time_out_call,
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
        ) = [pd.NA] * 18

        if df_text.shape[0] > 0:
            temp_text = df_text[
                (df_text["timestamp"] / 1000 >= stamp)
                & (df_text["timestamp"] / 1000 < stamp + step_size)
            ]
            m_len = np.array(temp_text["message length"])
            for k in range(len(m_len)):
                if m_len[k] == "MMS":
                    m_len[k] = 0
                if isinstance(m_len[k], str) is False:
                    if np.isnan(m_len[k]):
                        m_len[k] = 0
            m_len = m_len.astype(int)

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
            index_mms_r = (
                np.array(temp_text["sent vs received"]) == "received MMS"
            )
            num_s = sum(index_s.astype(int))
            num_r = sum(index_r.astype(int))
            num_mms_s = sum(index_mms_s.astype(int))
            num_mms_r = sum(index_mms_r.astype(int))
            total_char_s = sum(m_len[index_s])
            total_char_r = sum(m_len[index_r])
            if frequency == Frequency.DAILY:
                received_no_response = []
                sent_no_response = []
                # find the phone number in sent_from, but not in send_to
                for tel in receive_from_number:
                    if tel not in send_to_number:
                        received_no_response.append(tel)
                for tel in send_to_number:
                    if tel not in receive_from_number:
                        sent_no_response.append(tel)
                text_reciprocity_incoming = 0
                text_reciprocity_outgoing = 0
                for tel in received_no_response:
                    text_reciprocity_incoming = text_reciprocity_incoming
                    text_reciprocity_incoming += sum(
                        index_r *
                        (np.array(temp_text["hashed phone number"]) == tel)
                    )
                for tel in sent_no_response:
                    text_reciprocity_outgoing = text_reciprocity_outgoing
                    text_reciprocity_outgoing += sum(
                        index_s *
                        (np.array(temp_text["hashed phone number"]) == tel)
                    )

        if df_call.shape[0] > 0:
            temp_call = df_call[
                (df_call["timestamp"] / 1000 >= stamp)
                & (df_call["timestamp"] / 1000 < stamp + step_size)
            ]
            dur_in_sec = np.array(temp_call["duration in seconds"])
            dur_in_sec[np.isnan(dur_in_sec)] = 0
            dur_in_min = dur_in_sec / 60
            index_in_call = np.array(temp_call["call type"]) == "Incoming Call"
            index_out_call = (
                np.array(temp_call["call type"]) == "Outgoing Call"
            )
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
        if frequency == Frequency.DAILY:
            newline = [
                year,
                month,
                day,
                num_in_call,
                num_out_call,
                num_mis_call,
                num_uniq_in_call,
                num_uniq_out_call,
                num_uniq_mis_call,
                total_time_in_call,
                total_time_out_call,
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
            ]
        else:
            newline = [
                year,
                month,
                day,
                hour,
                num_in_call,
                num_out_call,
                num_mis_call,
                num_uniq_in_call,
                num_uniq_out_call,
                num_uniq_mis_call,
                total_time_in_call,
                total_time_out_call,
                num_s,
                num_r,
                num_mms_s,
                num_mms_r,
                num_s_tel,
                num_r_tel,
                total_char_s,
                total_char_r,
            ]
        summary_stats.append(newline)
    if frequency == Frequency.DAILY:
        stats_pdframe = pd.DataFrame(
            summary_stats,
            columns=[
                "year",
                "month",
                "day",
                "num_in_call",
                "num_out_call",
                "num_mis_call",
                "num_in_caller",
                "num_out_caller",
                "num_mis_caller",
                "total_mins_in_call",
                "total_mins_out_call",
                "num_s",
                "num_r",
                "num_mms_s",
                "num_mms_r",
                "num_s_tel",
                "num_r_tel",
                "total_char_s",
                "total_char_r",
                "text_reciprocity_incoming",
                "text_reciprocity_outgoing",
            ],
        )
    else:
        stats_pdframe = pd.DataFrame(
            summary_stats,
            columns=[
                "year",
                "month",
                "day",
                "hour",
                "num_in_call",
                "num_out_call",
                "num_mis_call",
                "num_in_caller",
                "num_out_caller",
                "num_mis_caller",
                "total_mins_in_call",
                "total_mins_out_call",
                "num_s",
                "num_r",
                "num_mms_s",
                "num_mms_r",
                "num_s_tel",
                "num_r_tel",
                "total_char_s",
                "total_char_r",
            ],
        )

    return stats_pdframe


# Main function/wrapper should take standard arguments with Beiwe names:
def log_stats_main(
    study_folder: str,
    output_folder: str,
    tz_str: str,
    frequency: Frequency,
    time_start=None,
    time_end=None,
    beiwe_id=None,
):
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    if frequency == Frequency.HOURLY_AND_DAILY:
        if os.path.exists(output_folder + "/hourly") is False:
            os.mkdir(output_folder + "/hourly")
        if os.path.exists(output_folder + "/daily") is False:
            os.mkdir(output_folder + "/daily")
    # beiwe_id should be a list of str
    if beiwe_id is None:
        beiwe_id = os.listdir(study_folder)
        id_w_folder = []
        for i in beiwe_id:
            if os.path.isdir(study_folder + "/" + i):
                id_w_folder.append(i)
        beiwe_id = id_w_folder

    if len(beiwe_id) > 0:
        for ID in beiwe_id:
            sys.stdout.write("User: " + ID + "\n")
            try:
                # read data
                text_data, text_stamp_start, text_stamp_end = read_data(
                    ID, study_folder, "texts", tz_str, time_start, time_end
                )
                call_data, call_stamp_start, call_stamp_end = read_data(
                    ID, study_folder, "calls", tz_str, time_start, time_end
                )
                if text_data.shape[0] > 0 or call_data.shape[0] > 0:
                    # stamps from call and text should be the stamp_end
                    sys.stdout.write("Data imported ..." + "\n")
                    stamp_start = min(text_stamp_start, call_stamp_start)
                    stamp_end = max(text_stamp_end, call_stamp_end)
                    # process data
                    if frequency == Frequency.HOURLY_AND_DAILY:
                        stats_pdframe1 = comm_logs_summaries(
                            ID,
                            text_data,
                            call_data,
                            stamp_start,
                            stamp_end,
                            tz_str,
                            Frequency.HOURLY,
                        )
                        stats_pdframe2 = comm_logs_summaries(
                            ID,
                            text_data,
                            call_data,
                            stamp_start,
                            stamp_end,
                            tz_str,
                            Frequency.DAILY,
                        )
                        write_all_summaries(
                            ID, stats_pdframe1, output_folder + "/hourly"
                        )
                        write_all_summaries(
                            ID, stats_pdframe2, output_folder + "/daily"
                        )
                    else:
                        stats_pdframe = comm_logs_summaries(
                            ID,
                            text_data,
                            call_data,
                            stamp_start,
                            stamp_end,
                            tz_str,
                            frequency,
                        )
                        write_all_summaries(ID, stats_pdframe, output_folder)
                    sys.stdout.write(
                        "Summary statistics obtained. Finished.\n"
                    )
            except:
                sys.stdout.write(
                    "An error occured when processing the data.\n"
                )
                pass
