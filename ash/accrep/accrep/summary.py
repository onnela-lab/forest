'''Summarize variables from raw Beiwe accelerometer files.

'''
import os
import logging
from beiwetools.helpers.time import (local_now, to_timestamp, 
                                     filename_time_format)
from beiwetools.helpers.log import log_to_csv
from beiwetools.helpers.functions import (setup_directories, setup_csv, 
                                          write_to_csv, summary_statistics)
from beiwetools.helpers.decorators import easy
from beiwetools.helpers.trackers import SamplingSummary, RangeTracker
from beiwetools.helpers.templates import ProcessTemplate
from .headers import summary_records_header, user_summary_header
from .functions import read_acc, metrics, transformations


logger = logging.getLogger(__name__)


@easy(['records_file', 'sampling_dir', 'data_dir'])
def setup_output(proc_dir, track_time):
    # set up directories
    out_dir = os.path.join(proc_dir, 'accrep', 'summary')
    if track_time:
        temp = local_now().replace(' ', '_')
        out_dir = os.path.join(out_dir, temp)    
    data_dir     = os.path.join(out_dir, 'data')
    sampling_dir = os.path.join(out_dir, 'sampling')
    records_dir  = os.path.join(out_dir, 'records')
    log_dir = os.path.join(out_dir, 'log')   
    setup_directories([out_dir, data_dir, records_dir, sampling_dir, log_dir])
    # set up records file
    records_file = setup_csv('records', records_dir, summary_records_header)
    # set up logging output
    log_to_csv(log_dir)
    logger.info('Created output directories.')
    return(records_file, sampling_dir, data_dir)        


@easy(['file_paths', 'trackers', 'user_file'])
def setup_user(user_id, project, intersample_cutoff_ms, data_dir,
               get_summaries, get_metrics):
    # get file paths
    file_paths = project.data[user_id].passive['accelerometer']['files']
    # set up trackers
    trackers = {}
    for ax in ['x', 'y', 'z']:
        trackers[ax] = RangeTracker()
    trackers['timestamp'] = SamplingSummary(intersample_cutoff_ms)
    # set up user output
    header = user_summary_header + \
             [s[1]+'_'+s[0] for s in get_summaries] + \
             get_metrics
    user_file = setup_csv(user_id, data_dir, header)
    logger.info('Finished setting up for user %s.' % user_id)
    return(file_paths, trackers, user_file)


@easy(['n_files', 'first_file', 'last_file', 'n_observations'])
def summarize_user(user_id, file_paths, trackers, do_transformations,
                   get_summaries, get_metrics, user_file, records_file,
                   sampling_dir):
    n_observations = 0
    for f in file_paths:
        try:
            # read file
            data = read_acc(f)
            n = len(data)
            n_observations += n
            timestamp = to_timestamp(os.path.basename(f).split['.'][0], 
                                     filename_time_format)
            elapsed_s = (data.timestamp[n-1] - data.timestamp[0])/1000
            to_write = [f, timestamp, n, elapsed_s]
            # update trackers
            for v in trackers:
                trackers[v].update(data[v])
            # do transformations
            for t in transformations:
                t.apply(data)
            # get summaries
            for s in get_summaries:
                function_name, variable = s                        
                function = summary_statistics[function_name]
                to_write.append(function(data[variable]))
            # get metrics
            for m in get_metrics:
                function = metrics[m]
                to_write.append(function(data))
            # write results
            write_to_csv(user_file, to_write)
        except:
            logger.warning('Unable to summarize file: %s' \
                           % os.path.basename(f))       
    # save intersample times
    trackers['timestamp'].save(sampling_dir, user_id)
    # write summary
    to_write = [user_id, len(file_paths), n_observations,
                trackers['timestamp'].frequency(),
                trackers['x'].stats['min'], trackers['x'].stats['max'],
                trackers['y'].stats['min'], trackers['y'].stats['max'],
                trackers['z'].stats['min'], trackers['z'].stats['max']]
    write_to_csv(records_file, to_write)
    logger.info('Finished summarizing files for user %s.' % user_id)
    return()


# Absolute threshold
# mPPZ, mZMO
# x sENMO, rENMO, 
# x sAD, rAD, 
# x sX, rX, 
# x sY, rY, 
# x sZ, rZ
# x sSA, rSA

# Relative
# mPPZ, mZMO,
# x sENMO, rENMO, 
# x sAD, rAD, 
# x sX, rX, 
# x sY, rY, 
# x sZ, rZ
# x sSA, rSA
# AI, AI0
# ARI, ARI0

all_transformations = ['vecmag', 'enmo', 'ampdev', 'sumamp', 'ppz', 'zmo']
all_summaries = [('mean', 'ppz'),   ('mean', 'zmo'), 
                 ('std', 'enmo'),   ('range', 'enmo'), 
                 ('std', 'ampdev'), ('range', 'ampdev'),
                 ('std', 'x'), ('range', 'x'), 
                 ('std', 'y'), ('range', 'y'),
                 ('std', 'z'), ('range', 'z'),
                 ('std', 'sumamp'), ('range', 'sumamp')]
all_metrics = ['ai', 'ai0', 'ari', 'ari0']

def pack_summary_kwargs(user_ids, proc_dir, project,
                        intersample_cutoff_ms,
                        do_transformations = all_transformations,
                        get_summaries = all_summaries,
                        get_metrics = [],



                        track_time = True, id_lookup = {}):
    '''
    Packs kwargs for gpsrep.Summary.do().
    
    Args:
        user_ids (list): List of identifiers (str).
        proc_dir (str): Path to folder where processed data can be written.
        project (beiwetools.manage.classes.BeiweProject):
            Representation of raw study data locations.
        intersample_cutoff_ms (int): Intersample times (milliseconds) above 
            this cutoff are ignored when summarizing the sampling rate.
            Something like 90% of accelerometer_off_duration may be reasonable.

        do_transformations (list):

        get_summaries (list):
            (<variable>, <summary statistic>)

        get_metrics (list):
            

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
        'intersample_cutoff_ms': intersample_cutoff_ms,
        'do_transformations': do_transformations,
        'get_summaries': get_summaries,
        'get_metrics': get_metrics,
        'track_time': track_time
        }
    kwargs['id_lookup'] = id_lookup
    return(kwargs)


Summary = ProcessTemplate.create(__name__,
                                [setup_output], [setup_user], 
                                [summarize_user], [], [])


