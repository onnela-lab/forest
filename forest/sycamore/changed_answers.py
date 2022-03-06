import datetime
from typing import Tuple, Any

import pandas as pd
import numpy as np

from forest.sycamore.functions import parse_surveys
from pandas import DataFrame


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
            The Final answer is in the 'last_answer' field
    """
    cols = ['survey id', 'beiwe_id', 'question id', 'question text',
            'question type', 'question index']

    agg['last_answer'] = agg.groupby(cols).answer.transform('last')
    # add in an answer ID and take the last of that too to join back on the
    # time
    agg = agg.reset_index().set_index(cols)
    agg['all_answers'] = agg.groupby(cols)['answer'].apply(lambda x: list(x))
    # Subset answer lists if needed
    agg['all_answers'] = agg['all_answers'].apply(
        lambda x: x if isinstance(x, float) else subset_answer_choices(x))
    agg['num_answers'] = agg['all_answers'].apply(
        lambda x: x if isinstance(x, float) else len(list(x)))
    agg = agg.reset_index()

    agg['first_time'] = agg.groupby(cols)['Local time'].transform('first')
    agg['last_time'] = agg.groupby(cols)['Local time'].transform('last')

    # Number of changed answers and time spent answering each question
    agg['time_to_answer'] = agg['last_time'] - agg['first_time']

    # Filter agg down to the line of the last question time
    agg = agg.loc[agg['Local time'] == agg['last_time']]

    return agg.reset_index(drop=True)


# Create a summary file that has survey, beiwe id, question id, average
# number of changed answers, average time spent answering question

def agg_changed_answers_summary(
        config_path: str, agg: pd.DataFrame) -> Tuple[Any, DataFrame]:
    """Create Summary File

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
            The Final answer is in the 'last_answer' field
        out(DataFrame):
            Summary of how an individual answered each question, with their
            most common answer, time to answer, etc
    """
    detail = agg_changed_answers(agg)
    #####################################################################
    # Add instance id and update first time to be the last last time if there
    # is only one line
    surv_config = parse_surveys(config_path)
    surv_config['q_internal_id'] = surv_config.groupby(
        'config_id').cumcount() + 1

    detail = pd.merge(detail,
                      surv_config[['question_id', 'q_internal_id']],
                      how='left',
                      left_on='question id',
                      right_on='question_id')

    detail['instance_id'] = np.where(detail['q_internal_id'] == 1, 1, 0)
    detail['instance_id'] = detail['instance_id'].cumsum()

    detail['last_time_1'] = detail.groupby(
        ['instance_id'])['last_time'].shift(1)

    # if there is one answer and the last_time and first_time are the same,
    # make the first_time = last_time_1
    detail['first_time'] = np.where(
        (detail.num_answers == 1)
        & (detail.first_time == detail.last_time)
        & (detail.q_internal_id != 1),
        detail.last_time_1,
        detail.first_time)

    # Update time_to_answer
    detail['time_to_answer'] = detail['last_time'] - detail['first_time']
    #####################################################################

    summary_cols = [
        'survey id',
        'beiwe_id',
        'question id',
        'question text',
        'question type']
    num_answers = detail.groupby(summary_cols)['num_answers'].count()
    avg_time = detail.groupby(summary_cols)['time_to_answer'].apply(
        lambda x: sum(x, datetime.timedelta()) / len(x)
    )
    avg_chgs = detail.groupby(summary_cols)['num_answers'].mean()

    def get_most_common_answer(col):
        answers = [answer for answer_list in col for answer in answer_list]
        return max(set(answers), key=answers.count, default=0)

    most_common_answer = detail.groupby(
        summary_cols)['all_answers'].apply(get_most_common_answer)

    out = pd.concat([num_answers, avg_time, avg_chgs,
                    most_common_answer], axis=1).reset_index()

    out.columns = summary_cols + ['num_answers',
                                  'average_time_to_answer',
                                  'average_number_of_answers',
                                  'most_common_answer']

    # Select relevant columns from detail to output and keep one line per
    # question
    detail = detail.loc[detail['last_answer'] == detail['answer']]
    detail_cols = [
        'survey id',
        'beiwe_id',
        'question id',
        'question text',
        'question type',
        'question answer options',
        'timestamp',
        'Local time',
        'last_answer',
        'all_answers',
        'num_answers',
        'first_time',
        'last_time',
        'time_to_answer']

    detail = detail[detail_cols]
    return detail, out
