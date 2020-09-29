'''Summarize variables from raw Beiwe GPS files.

'''
import os
import logging
from beiwetools.helpers.time import local_now
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (setup_directories, setup_csv, 
                                          write_to_csv)
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.trackers import (HistogramTracker, SamplingSummary, 
                                         RangeTracker)
from beiwetools.helpers.templates import ProcessTemplate
from .headers import raw_header, user_summary, variable_ranges
from .functions import read_gps


logger = logging.getLogger(__name__)


@easy(['out_file', 'accuracy_dir', 'sampling_dir'])
def setup_output(proc_dir, track_time, accuracy_percentiles):
    # setup directories
    out_dir = os.path.join(proc_dir, 'gps', 'summary')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir = os.path.join(out_dir, 'data')
    log_dir = os.path.join(out_dir, 'log')   
    accuracy_dir = os.path.join(out_dir, 'accuracy')
    sampling_dir = os.path.join(out_dir, 'sampling')
    setup_directories([out_dir, data_dir, log_dir, accuracy_dir, sampling_dir])
    # setup output file
    prefixes = ['accuracy_approx_', 'accuracy_closest_']
    percentiles_header = [s+str(p) for p in accuracy_percentiles 
                          for s in prefixes]
    header = user_summary + variable_ranges + percentiles_header
    out_file = setup_csv('records', data_dir, header)    
    # set up logging output
    log_to_csv(log_dir)
    logger.info('Created output directories.')
    return(out_file, accuracy_dir, sampling_dir)        


@easy(['file_paths', 'range_trackers', 'accuracy_histogram', 
       'sampling_summary'])
def setup_user(user_id, project, intersample_cutoff_ms):
    # get file paths
    file_paths = project.data[user_id].passive['gps']['files']
    # setup trackers
    range_trackers = {}
    for v in raw_header[2:]:
        range_trackers[v] = RangeTracker()
    accuracy_histogram = HistogramTracker.create(list(range(0, 100, 5)))
    sampling_summary = SamplingSummary(intersample_cutoff_ms)
    logger.info('Finished setting up for user %s.' % user_id)
    return(file_paths, range_trackers, accuracy_histogram, sampling_summary)


@easy(['n_files', 'first_file', 'last_file', 'n_observations'])
def summarize_user(user_id, file_paths, range_trackers,
                   accuracy_histogram, sampling_summary):
    n_observations = 0
    for f in file_paths:
        try:
            data = read_gps(f)
            n_observations += len(data)
            for v in range_trackers:
                range_trackers[v].update(data[v])
            accuracy_histogram.update(data.accuracy)
            sampling_summary.update(data.timestamp)
        except:
            logger.warning('Unable to summarize variables: %s' \
                           % os.path.basename(f))    
    n_files = len(file_paths)
    first_file = os.path.basename(file_paths[0])
    last_file = os.path.basename(file_paths[-1])
    logger.info('Finished summarizing variables for user %s.' % user_id)
    return(n_files, first_file, last_file, n_observations)


@easy([])
def write_user(user_id, range_trackers, accuracy_histogram, sampling_summary,
               accuracy_percentiles, n_files, first_file, last_file, 
               n_observations, out_file, accuracy_dir, sampling_dir):    
    # assemble summary
    summary_1 = [user_id, n_files, first_file, last_file, n_observations,
                 sampling_summary.frequency()]
    summary_2 = [None] * 2 * len(range_trackers)
    summary_2[::2] = [range_trackers[v].stats['min'] for v in raw_header[2:]]
    summary_2[1::2] = [range_trackers[v].stats['max'] for v in raw_header[2:]]
    summary_3 = []
    for p in accuracy_percentiles:
        approx, closest = accuracy_histogram.approx_percentile(p/100)
        summary_3 += [approx, closest]
    # write summary
    write_to_csv(out_file, summary_1 + summary_2 + summary_3)
    # save objects
    sampling_summary.save(sampling_dir, user_id)
    accuracy_histogram.export(user_id, accuracy_dir)
    logger.info('Finished writing summaries for user %s.' % user_id)


def pack_summary_kwargs(user_ids, proc_dir, project,
                        accuracy_percentiles,
                        intersample_cutoff_ms,
                        track_time = True, id_lookup = {}):
    '''
    Packs kwargs for gpsrep.Summary.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        accuracy_percentiles (list): List of percentiles.  Enter each
            percentile as a number between 0 and 100, e.g. 95 for the 95th
            percentile.            
        intersample_cutoff_ms (int): Intersample times (milliseconds) above 
            this cutoff are ignored when summarizing the sampling rate.
            Something like 90% of gps_off_duration may be reasonable.
        track_time (bool): If True, output to a timestamped folder.    
        id_lookup (dict): Optional.
            If identifiers in user_ids aren't Fitabase identifiers,
            then this should be a dictionary in which:
                - keys are identifiers,
                - values are the corresponding Fitabase identifers.

    Returns:
        kwargs (dict):
            Packed keyword arguments.
            To run a summary: Summary.do(**kwargs)
    '''
    kwargs = {}
    kwargs['user_ids'] = user_ids    
    kwargs['process_kwargs'] = {
        'proc_dir': proc_dir,
        'project': project,
        'accuracy_percentiles': accuracy_percentiles,
        'intersample_cutoff_ms': intersample_cutoff_ms,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Summary = ProcessTemplate.create(__name__,
                                [setup_output], [setup_user], 
                                [summarize_user], [write_user], [])


