''' Workflow for processing raw Beiwe power state data with powrep.

'''
###############################################################################
# 0. Set up paths
###############################################################################
raw_dir  = 'path/to/directory/containing/raw/Beiwe/user/data'
proc_dir = 'path/to/directory/for/powrep/output'

###############################################################################
# 1. Create a BeiweProject
###############################################################################
from beiwetools.manage import BeiweProject
p = BeiweProject.create(raw_dir)

###############################################################################
# 2. Summarize power state data
# - This is mainly for preliminary data exploration and troubleshooting.
###############################################################################
from powrep import pack_summary_kwargs, Summary
# pack arguments for Summary.do():
kwargs = pack_summary_kwargs(user_ids = p.ids, # summarize data for all 
                                               # available Beiwe IDs
                             proc_dir = proc_dir, 
                             project = p, 
                             track_time = True # isolate output in a 
                                               # timestamped directory
                                               )
# run the summary:
Summary.do(**kwargs)

# Now review proc_dir/summary/log/log.csv:
#   - Check for log records with "WARNING" in the levelname column.
#   - Important warning messages include:
#       - "Header is not recognized"
#       - "Unrecognized event"
#       - "Unable to summarize"

# Data summaries are found in proc_dir/summary/records/records.csv:
#   - The first eight columns contain summaries that are common across 
#     platforms, e.g. file counts and observation counts.
#   - The next six columns contain iOS-specific event counts, and should 
#     not contain records for Android devices.
#   - The last ten columns contain Android-specific event counts.

###############################################################################
# 3. Extract events by category
# - 
###############################################################################
from powrep import pack_extract_kwargs, Extract


# pack arguments for Extract.do():
kwargs = pack_extract_kwargs(user_ids = r.ids, 
                             proc_dir = proc_dir, 
                             project = p, 
                             track_time = True)
