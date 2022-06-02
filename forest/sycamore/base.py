import logging
import os
from typing import Optional, List

from forest.constants import Frequency
from forest.utils import get_ids
from forest.sycamore.common import (aggregate_surveys_config,
                                    aggregate_surveys_no_config,
                                    get_month_from_today
                                    )
from forest.sycamore.constants import EARLIEST_DATE
from forest.sycamore.responses import (agg_changed_answers_summary,
                                       format_responses_by_submission)
from forest.sycamore.submits import (survey_submits,
                                     survey_submits_no_config,
                                     summarize_submits)

logger = logging.getLogger(__name__)


def compute_survey_stats(
        study_folder: str, output_folder: str, tz_str: str = "UTC",
        users: Optional[List] = None,
        start_date: str = EARLIEST_DATE, end_date: Optional[str] = None,
        config_path: Optional[str] = None, interventions_filepath: str = None,
        augment_with_answers: bool = True, submits_timeframe: str = None,
        submits_by_survey_id: bool = True
) -> bool:
    """Compute statistics on surveys

    Args:
    output_folder:
        File path to output summaries and details
    study_folder:
        File path to study data
    config_path:
        File path to study configuration file
    start_date:
        The earliest date of survey data to read in, in YYYY-MM-DD format
    end_date:
        The latest survey data to read in, in YYYY-MM-DD format
    users:
        List of users in study for which we are generating a survey schedule
    tz_str:
        Timezone of study. This defaults to "UTC"
    interventions_filepath:
        filepath where interventions json file is.
    augment_with_answers:
        Whether to use the survey_answers stream to fill in missing surveys
        from survey_timings
    submits_timeframe:
        The timeframe to summarize survey submissions over. One of
        "both", "daily", or "hourly". An overall summary for each user is
        always generated ("submits_summary_overall.csv"), and submissionns can
        also be generated across days ("submits_summary_daily.csv"), hours
        ("submits_summary_hourly.csv") or both.
    submits_by_survey_id:
        Summmarize survey submits with separate lines for different
        surveys in submits_summary.csv. By default, this is True, so a
        different line for each survey will be generated.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "summaries"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "by_survey"), exist_ok=True)
    if users is None:
        users = get_ids(study_folder)
    if end_date is None:
        end_date = get_month_from_today()
    # Read, aggregate and clean data
    if config_path is None:
        logger.warning("No config file provided. "
                       "Skipping some summary outputs.")
        agg_data = aggregate_surveys_no_config(
            study_folder, tz_str, users, start_date, end_date,
            augment_with_answers
        )
        if agg_data.shape[0] == 0:
            logger.error("Error: No survey data found in %s", study_folder)
            return True
    else:
        agg_data = aggregate_surveys_config(
            study_folder, config_path, tz_str, users, start_date,
            end_date,  augment_with_answers
        )
        if agg_data.shape[0] == 0:
            logger.error("Error: No survey data found in %s", study_folder)
            return True
        # Create changed answers detail and summary
        ca_detail, ca_summary = agg_changed_answers_summary(config_path,
                                                            agg_data)
        ca_detail.to_csv(
            os.path.join(output_folder, "summaries", "answers_data.csv"),
            index=False
        )
        ca_summary.to_csv(
            os.path.join(output_folder, "summaries", "answers_summary.csv"),
            index=False
        )
        if start_date is not None and end_date is not None:
            # Create survey submits detail and summary
            ss_detail = survey_submits(
                config_path, start_date, end_date,
                users, agg_data, interventions_filepath
            )
            ss_summary = summarize_submits(ss_detail, None,
                                           submits_by_survey_id)
            if ss_summary.shape[0] > 0:
                ss_detail.to_csv(os.path.join(output_folder, "summaries",
                                              "submits_and_deliveries.csv"),
                                 index=False)
                ss_summary.to_csv(
                    os.path.join(output_folder, "summaries",
                                 "submits_summary.csv"), index=False
                )
                if Frequency(submits_timeframe) == Frequency.BOTH:
                    ss_summary_h = summarize_submits(
                        ss_detail, Frequency.HOURLY, submits_by_survey_id
                    )
                    ss_summary_d = summarize_submits(
                        ss_detail, Frequency.DAILY, submits_by_survey_id
                    )
                    ss_summary_d.to_csv(
                        os.path.join(output_folder, "summaries",
                                     "submits_summary_daily.csv"), index=False
                    )
                    ss_summary_h.to_csv(
                        os.path.join(output_folder, "summaries",
                                     "submits_summary_hourly.csv"), index=False
                    )
                elif Frequency(submits_timeframe) == Frequency.HOURLY:
                    ss_summary_h = summarize_submits(
                        ss_detail, Frequency.HOURLY, submits_by_survey_id
                    )
                    ss_summary_h.to_csv(
                        os.path.join(output_folder, "summaries",
                                     "submits_summary_hourly.csv"), index=False
                    )
                elif Frequency(submits_timeframe) == Frequency.DAILY:
                    ss_summary_d = summarize_submits(
                        ss_detail, Frequency.DAILY, submits_by_survey_id
                    )
                    ss_summary_d.to_csv(
                        os.path.join(output_folder, "summaries",
                                     "submits_summary_daily.csv"), index=False
                    )
            else:
                logger.error("An Error occurred when "
                             "getting survey submit summaries")
    surveys_dict = format_responses_by_submission(agg_data)
    for survey_id in surveys_dict.keys():
        surveys_dict[survey_id].to_csv(
            os.path.join(output_folder, "by_survey", survey_id + ".csv"),
            index=False
        )

    # Write out summaries
    agg_data.to_csv(
        os.path.join(output_folder, "summaries", "agg_survey_data.csv"),
        index=False
    )
    # Add alternative survey submits table
    submits_tbl = survey_submits_no_config(agg_data)
    submits_tbl.to_csv(
        os.path.join(output_folder, "summaries", "submits_only.csv"),
        index=False
    )
    return True
