''' Workflow for processing Fitabase data with fitrep.

'''
###############################################################################
# 0. Set up paths
###############################################################################
fitabase_dir = 'path/to/directory/containing/individual/Fitabase/files'
proc_dir = 'path/to/directory/for/fitrep/output'

###############################################################################
# 1. Choose file types
###############################################################################
# This pacakge can process any file type in raw_header.keys().
# To process all file types:
from fitrep import raw_header
file_types = list(raw_header.keys())

###############################################################################
# 2. Create a registry of the Fitabase directory
###############################################################################
from fitrep import FitabaseRegistry
r = FitabaseRegistry.create(fitabase_dir)

###############################################################################
# 3. Summarize directory contents
# - This is mainly for preliminary data exploration and troubleshooting.
###############################################################################
from fitrep import pack_summary_kwargs, Summary
# pack arguments for Summary.do():
kwargs = pack_summary_kwargs(user_ids = r.ids, # summarize data for all 
                                               # available Fitabase IDs
                             proc_dir = proc_dir, 
                             file_types = file_types, 
                             registry = r, 
                             track_time = True # isolate output in a 
                                               # timestamped directory
                                               )
# run the summary:
Summary.do(**kwargs)

# Now review proc_dir/fitrep/summary/<timestamp>/log/log.csv:
#   - Check for log records with "WARNING" in the levelname column.
#   - Important warning messages include:
#       - "Contains smartphone app data"
#       - "Unknown device"
#       - "Unknown service provider"

# Data summaries are found in proc_dir/fitrep/summary/<timestamp>/records/<file_type>.csv.
#   - Review /records/syncEvents.csv:
#       - Identify any unexpected entries, such as unknown devices, short 
#         followup periods, or absence of sync events. 
#   - Review /records/minuteSleep.csv:
#       - Note p_offsets, the proportion of sleep observations with a 30-second
#         synchronization offset.

# At this point, it may be desirable to define a restricted set of Fitabase ids,
# file types, or followup periods.  
#   - A Fitabase ID with data mingled from multiple sources (devices or apps)
#     may need to be excluded.
#   - See documentation on format for followup_ranges, which can be passed 
#     to fitrep.Sync.do() and fitrep.Format.do().
#   - Also see documentation on format for id_lookup, an optional argument
#     for using alternate identifiers.

###############################################################################
# 4. Process synchronization events
# - Get some summary statistics on the frequency of sync events and the 
#   number of timezones (UTC offsets) experienced by the device.
###############################################################################
from fitrep import pack_sync_kwargs, Sync
# pack arguments for Sync.do():
kwargs = pack_sync_kwargs(user_ids = r.ids, 
                          proc_dir = proc_dir, 
                          registry = r, 
                          track_time = True)
# process synchronization events:
Sync.do(**kwargs)

# A FutureWarning may originate from pandas due to fitrep.functions.py line 88.
#   - This is due to a documented issue with Numpy.  See:
#     https://www.thetopsites.net/article/50997927.shtml
#   - Probably not something to be concerned about, as long as no additional
#     warnings are emitted.

# Review proc_dir/fitrep/sync/<timestamp>/log/log.csv:
#   - Check for log records with "WARNING" in the levelname column.
#   - Important warning messages include references to incomplete/missing
#     device logs, or multiple device logs.  
#   - Proceed with caution if device logs include syncs with both Fitbit 
#     devices AND the Fitbit smartphone app.  This package can't separate 
#     out mingled data from both sources.

# Sync summaries are in proc_dir/fitrep/sync/<timestamp>/records/records.csv.
#   - Users with low values in the n_syncs column may have poor data quality.
#   - Check for any non-zero values in the n_app_syncs column.
#   - User-level intersync times (in seconds) are summarized with min/max, 
#     mean, and median.
#   - Global intersync times are summarized in the last row of this csv.
#   - A reasonable median time between syncs may be around 900-2000 seconds.

###############################################################################
# 5. Reformat data to a Beiwe-friendly format
# - Convert Fitabase date-times to Beiwe date-times.
# - Re-synchronize Fitbit sleep classifications.
###############################################################################
from fitrep import pack_format_kwargs, Format
# pack arguments for Format.do():
kwargs = pack_format_kwargs(user_ids = r.ids,
                            proc_dir = proc_dir,
                            file_types = file_types, 
                            registry = r, 
                            track_time = True)
    
# reformat files:
Format.do(**kwargs)    
    
# Review proc_dir/fitrep/format/<timestamp>/log/log.csv:
#   - Check for log records with "WARNING" in the levelname column.
#   - Warning messages may correspond to specific files that could not be 
#     processed.

# Review individual records in proc_dir/fitrep/format/<timestamp>/records/.
#   - Check follow up periods for each combination of user & file type.

# Reformatted files are in proc_dir/fitrep/format/<timestamp>/data/.
#   - Column headers are "local_datetime" and "value".
#   - minuteSleep files have another column, "resync".  If resync = False, the
#     observation is an original Fitbit sleep classification.  An observation 
#     is tagged resync = True if it was assembled from two neighboring 
#     observations that agreed on sleep classification.
