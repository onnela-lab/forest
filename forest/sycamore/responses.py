import datetime
from typing import Tuple, Any
import logging

import pandas as pd
from pandas import DataFrame
import numpy as np

from forest.sycamore.common import parse_surveys


logger = logging.getLogger(__name__)


def subset_answer_choices(answer: list) -> list:
    """Remove Duplicate Answers

    If a user changes their answers multiple times, an iOS device will have
    redundant answers at the beginning and end of the list, so we remove them.
    Args:
        answer(list):
            List of changed answers

    Returns:
        answer(list):
            List of changed answers with redundant answers removed
    """
    if isinstance(answer[0], float):
        answer = answer[1:]

    if len(answer) > 1:
        if answer[-1] == answer[-2]:
            answer = answer[:-1]
    return answer


def agg_changed_answers(agg: pd.DataFrame) -> pd.DataFrame:
    """Add first/last times and changed answers to agg DataFrame

    Function that takes aggregated data and adds list of changed answers and
    first and last times and answers.

    Args:
        agg(DataFrame):
            Aggregated Data


    Returns:
        agg(DataFrame):
            Dataframe with aggregated data, one line per question answered,
            with changed answers aggregated into a list.
            The Final answer is in the "last_answer" field
    """
    cols = ["survey id", "beiwe_id", "question id", "question text",
            "question type", "question index"]

    agg["last_answer"] = agg.groupby(cols).answer.transform("last")
    # add in an answer ID and take the last of that too to join back on the
    # time
    agg = agg.reset_index().set_index(cols)
    agg["all_answers"] = agg.groupby(cols)["answer"].apply(lambda x: list(x))
    # Subset answer lists if needed
    agg["all_answers"] = agg["all_answers"].apply(
        lambda x: x if isinstance(x, float) else subset_answer_choices(x)
    )
    agg["num_answers"] = agg["all_answers"].apply(
        lambda x: x if isinstance(x, float) else len(list(x))
    )
    agg = agg.reset_index()

    agg["first_time"] = agg.groupby(cols)["Local time"].transform("first")
    agg["last_time"] = agg.groupby(cols)["Local time"].transform("last")

    # Number of changed answers and time spent answering each question
    agg["time_to_answer"] = agg["last_time"] - agg["first_time"]

    # time to answer is meaningless if the result came from survey_answers,
    # where all questions in a survey have the same time.
    agg["time_to_answer"] = np.where(
        agg["data_stream"] == "survey_answers", agg["time_to_answer"],
        np.datetime64('NaT')
    )
    # Filter agg down to the line of the last question time
    agg = agg.loc[agg["Local time"] == agg["last_time"]]

    return agg.reset_index(drop=True)


def agg_changed_answers_summary(
        config_path: str, agg: pd.DataFrame
) -> Tuple[Any, DataFrame]:
    """Create Summary File with survey, beiwe id, question id, average
    number of changed answers, average time spent answering question

    Args:
        config_path(str):
            File path to study configuration file
        agg(DataFrame):
            Dataframe with aggregated data (output from
            aggregate_surveys_config)


    Returns:
        detail(DataFrame):
            Dataframe with aggregated data, one line per question answered,
            with changed answers aggregated into a list.
            The Final answer is in the "last_answer" field
        out(DataFrame):
            Summary of how an individual answered each question, with their
            most common answer, time to answer, etc
    """
    detail = agg_changed_answers(agg)
    #####################################################################
    # Add instance id and update first time to be the last time if there
    # is only one line
    surv_config = parse_surveys(config_path)
    surv_config["q_internal_id"] = surv_config.groupby(
        "config_id"
    ).cumcount() + 1

    detail = pd.merge(
        detail, surv_config[["question_id", "q_internal_id"]], how="left",
        left_on="question id", right_on="question_id"
    )

    detail["instance_id"] = np.where(detail["q_internal_id"] == 1, 1, 0)
    detail["instance_id"] = detail["instance_id"].cumsum()

    detail["last_time_1"] = detail.groupby(
        ["instance_id"]
    )["last_time"].shift(1)

    # if there is one answer and the last_time and first_time are the same,
    # make the first_time = last_time_1
    detail["first_time"] = np.where(
        (detail.num_answers == 1)
        & (detail.first_time == detail.last_time)
        & (detail.q_internal_id != 1),
        detail.last_time_1, detail.first_time
    )

    # Update time_to_answer
    detail["time_to_answer"] = detail["last_time"] - detail["first_time"]

    # note that "time to answer is meaningless for answers datastreams because
    # each question in a survey will have exactly the same time.
    detail["time_to_answer"] = np.where(
        detail["data_stream"] == "survey_answers",
        detail["time_to_answer"],
        np.datetime64('NaT')
    )

    #####################################################################

    summary_cols = ["survey id", "beiwe_id", "question id", "question text",
                    "question type"]
    num_answers = detail.groupby(summary_cols)["num_answers"].count()
    avg_time = detail.groupby(summary_cols)["time_to_answer"].apply(
        lambda x: sum(x, datetime.timedelta()) / len(x)
    )
    avg_chgs = detail.groupby(summary_cols)["num_answers"].mean()

    def get_most_common_answer(col):
        answers = [answer for answer_list in col for answer in answer_list]
        return max(set(answers), key=answers.count, default=0)

    most_common_answer = detail.groupby(
        summary_cols
    )["all_answers"].apply(get_most_common_answer)

    out = pd.concat([num_answers, avg_time, avg_chgs,
                    most_common_answer], axis=1).reset_index()

    out.columns = summary_cols + [
        "num_answers", "average_time_to_answer", "average_number_of_answers",
        "most_common_answer"
    ]

    # Select relevant columns from detail to output and keep one line per
    # question
    detail = detail.loc[detail["last_answer"] == detail["answer"]]
    detail_cols = ["survey id", "beiwe_id", "question id", "question text",
                   "question type", "question answer options", "timestamp",
                   "Local time", "last_answer", "all_answers", "num_answers",
                   "first_time", "last_time", "time_to_answer"]

    detail = detail[detail_cols]
    return detail, out


def responses_by_submission(agg_data: pd.DataFrame) -> dict:
    """
    Takes aggregated answers and writes individual csv files for each survey

    Args:
        agg_data(DataFrame): output from aggregate_surveys_config or
            aggregate_surveys_no_config

    Returns:
        surveys_dict(dict): Dict with a key for each survey ID in agg_data.
        Each value is a dataframe with readable survey submission information.

    """
    surveys_dict = {}

    for survey_id in agg_data["survey id"].unique():
        logger.info("now processing: %s", survey_id)
        survey_df = agg_data.loc[agg_data["survey id"] == survey_id].copy()
        unique_survey_cols = ["beiwe_id", "surv_inst_flg"]

        # get starting and ending times for each survey
        survey_df["start_time"] = survey_df.groupby(unique_survey_cols)[
            "Local time"
        ].transform("first")
        survey_df["end_time"] = survey_df.groupby(unique_survey_cols)[
            "Local time"
        ].transform("last")

        survey_df = survey_df.loc[survey_df["submit_line"] != 1]
        # needs to come after finding start and end times because
        # the "user hit submit" line contains end times

        if survey_df.shape[0] > 0:  # We need to have data to read
            survey_df.sort_values(by=["beiwe_id", "Local time"],
                                  ascending=True, inplace=True)
            survey_df.reset_index(inplace=True)
            # TODO: add an option to take the survey config file to get this
            # information (must be done after survey ID is in the config file)
            id_text_dict = {}.fromkeys(survey_df["question id"].unique())

            num_found = 0
            i = 0
            keys_not_found = True
            while keys_not_found:
                # update dictionary entry
                if id_text_dict[survey_df.loc[i, "question id"]] is None:
                    id_text_dict[survey_df.loc[i, "question id"]] = [
                        survey_df.loc[i, "question text"],
                        survey_df.loc[i, "question type"],
                        survey_df.loc[i, "question answer options"]
                    ]
                    num_found = num_found + 1
                if num_found == len(id_text_dict):
                    keys_not_found = False
                i = i + 1
                if i == survey_df.shape[0]:
                    # should find all keys before we get to the end but let's
                    # be safe...
                    break

            survey_df["survey_duration"] = survey_df["end_time"] - survey_df[
                "start_time"
            ]

            survey_df["survey_duration"] = np.where(
                survey_df["data_stream"] == "survey_answers",
                survey_df["survey_duration"],
                np.datetime64('NaT')
            )

            keep_cols = ["beiwe_id", "start_time", "end_time",
                         "survey_duration"]

            unique_question_cols = keep_cols + ["question id"]
            survey_df.drop_duplicates(unique_question_cols, keep="last",
                                      inplace=True)
            # Because we sorted ascending, this will keep the most recent
            # response
            pivot_df = survey_df.pivot(index=keep_cols, columns="question id",
                                       values="answer")
            question_info_df = pd.DataFrame(id_text_dict)

            question_id_df = pd.DataFrame(columns=question_info_df.columns)
            # move column names to a row for writing
            question_id_df.loc[0] = question_info_df.columns
            # add fake indices to stack nicely with the multiindex
            for col in keep_cols[1:4]:
                question_info_df[col] = ""
                question_id_df[col] = ""

            question_id_df["beiwe_id"] = "Question ID"
            question_info_df["beiwe_id"] = ["Question Text", "Question Type",
                                            "Question Options"]
            # Get these to stack nicely with multiindex
            question_info_df.set_index(keys=keep_cols, inplace=True)
            question_id_df.set_index(keys=keep_cols, inplace=True)
            # stack together
            output_df = pd.concat([question_info_df, question_id_df, pivot_df])
            output_df = output_df.reset_index(drop=False)
            # Interpretable column names in csv
            colnames = ["beiwe_id", "start_time", "end_time",
                        "survey_duration"]
            colnames = colnames + [f"question_{i + 1}" for i in range(
                len(output_df.columns) - len(colnames))]
            output_df.columns = colnames

            surveys_dict[survey_id] = output_df

    return surveys_dict
