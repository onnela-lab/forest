import os
from typing import Optional, List
import logging

from forest.sycamore.responses import (agg_changed_answers_summary,
                                       responses_by_submission)
from forest.sycamore.common import (aggregate_surveys_config,
                                    aggregate_surveys_no_config)
from forest.sycamore.submits import (survey_submits,
                                     survey_submits_no_config,
                                     get_all_interventions_dict)

logger = logging.getLogger(__name__)


def survey_stats_main(
        study_folder: str, output_folder: str, tz_str: str = "UTC",
        participant_ids: Optional[List] = None,
        time_start: Optional[str] = None, time_end: Optional[str] = None,
        config_path: Optional[str] = None,
        interventions_filepath: Optional[str] = None,
        augment_with_answers: bool = True
) -> bool:
    """Compute statistics on surveys

    Args:
    output_folder(str):
        File path to output summaries and details
    study_folder(str):
        File path to study data
    config_path(str):
        File path to study configuration file
    time_start(str):
        The first date of the survey data
    time_end(str):
        The last date of the survey data
    participant_ids(list):
        List of users in study for which we are generating a survey schedule
    tz_str(str):
        Timezone of study. This defaults to "UTC"
    interventions_filepath(str):
        filepath where interventions json file is.
    augment_with_answers(bool):
        Whether to use the survey_answers stream to fill in missing surveys
        from survey_timings

    """
    os.makedirs(output_folder, exist_ok=True)
    if participant_ids is None:
        participant_ids = [u
                           for u in os.listdir(study_folder)
                           if not u.startswith(".") and u != "registry"]
    # Read, aggregate and clean data
    if config_path is None:
        logger.warning("No config file provided. "
                       "Skipping some summary outputs.")
        agg_data = aggregate_surveys_no_config(
            study_folder, tz_str, users,
            augment_with_answers = augment_with_answers
        )
        if agg_data.shape[0] == 0:
            logger.error("Error: No survey data found in %s", study_folder)
            return True
    else:
        agg_data = aggregate_surveys_config(
            study_folder, config_path, tz_str, users,
            augment_with_answers = augment_with_answers
        )
        if agg_data.shape[0] == 0:
            logger.error("Error: No survey data found in %s", study_folder)
            return True
        # Create changed answers detail and summary
        ca_detail, ca_summary = agg_changed_answers_summary(config_path,
                                                            agg_data.copy())
        ca_detail.to_csv(os.path.join(output_folder, "answers_data.csv"),
                         index=False)
        ca_summary.to_csv(os.path.join(output_folder, "answers_summary.csv"),
                          index=False)
        if time_start is not None and time_end is not None:
            # Create survey submits detail and summary
            all_interventions_dict = get_all_interventions_dict(
                interventions_filepath
            )
            ss_detail, ss_summary = survey_submits(
                config_path, time_start, time_end,
                participant_ids, agg_data.copy(), all_interventions_dict
            )
            if ss_summary.shape[0] > 0:
                ss_detail.to_csv(
                    os.path.join(output_folder, "submits_data.csv"),
                    index=False
                )
                ss_summary.to_csv(
                    os.path.join(output_folder, "submits_summary.csv"),
                    index=False
                )
            else:
                logger.error("An Error occurred when "
                             "getting survey submit summaries")

    surveys_dict = responses_by_submission(agg_data.copy())
    for survey_id in surveys_dict.keys():
        surveys_dict[survey_id].to_csv(
            os.path.join(output_folder, "by_survey", survey_id + ".csv"),
            index=False
        )

    # Write out summaries
    agg_data.to_csv(os.path.join(output_folder, "agg_survey_data.csv"),
                    index=False)
    # Add alternative survey submits table
    submits_tbl = survey_submits_no_config(agg_data.copy())
    submits_tbl.to_csv(os.path.join(output_folder, "submits_alt_summary.csv"),
                       index=False)
    return True
