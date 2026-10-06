# Willow

## Usage
Use the Willow tree to create summaries of the Calls and Texts data streams out of the Beiwe Android
App. (This data stream is restricted to the Android Platform, Apple does not provide access to this
information.)

The main function looks like this:

```python
from forest.constants import Frequency
from forest.willow import log_stats_main

log_stats_main(
     study_folder:   str,
     output_folder:  str,
     tz_str:         str,
     frequency:      Frequency,
     # Optional parameters
     time_start:     list | None = None,
     time_end:       list | None = None,
     beiwe_ids:      list[str] | None = None,
)
```

Output consists of a csv file for each subject in the study folder, with their participant ID as the
filename and each summary statistic about communication logs as a column. The statistics are
summarized in a daily/hourly manner based on your choice.

The input study folder uses the standard Beiwe folder structure, in this case a folder for each
participant with a subfolder containing the `calls` and `texts` folders. (Willow will read what is
available.)

- `study_folder`: string, the path of the study folder. The study folder should contain individual
  participant folders with subfolders `calls`,`texts` inside
- `output_folder`: string, the path of the folder where you want to save results
- `tz_str`: string, the standard timezone name where the study was based conducted. For example,
  "America/New_York".
- `frequency`: Use an option on the Frequency enum for the temporal resolution of summary statistics.
- `time_start` and `time_end` are starting time and ending time of the window of interest.
     -The time should be a list of integers with format [year, month, day, hour, minute, second] (default: None).
     - If `time_start` is None and `time_end` is None: then it reads all the available files.
     - If `time_start` is None and `time_end` is given, then it reads all the files before the given time.
     - If `time_start` is given and `time_end` is None, then it reads all the files after the given time.
- `beiwe_ids`: a list of Beiwe IDs. Provide participant ID strings if you wish to restrict your run
  of Willow to only those participants. Leave as None to include all participants.

___


## Summary Statistics Generated

|     Variable                          	|     Type     	|     Description of Variable                                                                                 	|
|---------------------------------------	|--------------	|-------------------------------------------------------------------------------------------------------------	|
|     year                      	|       int       	|     Year that observation was collected                                                   	|
|     month                	|              	|     Month of year that observation was collected                                 	|
|     day                             	|       int       	|     Day of Month of Year that observation was collected                                                     	|
|     hour |              	|     Hour of day that observation was collected (excluded if entries are computed on a daily level)                                                      	|
|     num_in_call                      	|        int      	|     Total number of incoming calls                                       	|
|     num_out_call                 	|        int      	|     Total number of outgoing calls                                               	|
|     num_mis_call               	|      int        	|     Total number of missed calls
|     num_in_caller            	|       float       	|     Total number of unique incoming callers                                         	|
|     num_out_caller          	|        int      	|     Total number of unique outgoing calls                                          	|
|     num_mis_caller      	|        float      	|     Total number of unique callers missed                                   	|
|     total_time_in_call                   	|      int        	|     Total amount of minutes spent on incoming calls                                              	|
|     total_time_out_call          	|       int       	|     Total amount of minutes spent on outgoing calls                                                      	|
|     num_uniq_individuals_call_or_text          	|       float       	|     Total number of unique individuals who called or texted the Beiwe user, or who the Beiwe user called or texted. The total number of individuals with any communication contact with the Beiwe user                                                      	|
|     num_s     	|        float      	|     Total number of sent SMS texts                                   	|
|     num_r                    	|      int        	|     Total number of received SMS texts                                                	|
|     num_mms_s                   	|      int        	|     Total number of sent MMS texts     	|
|     num_mms_r               	|        float      	|     Total number of received MMS texts   |
|     num_s_tel |      int        	|     Total number of people who received texts from subject |
|     num_r_tel|      int        	|     Total number of people who sent texts to subject   	|
|     total_char_s                  	|      int        	|     Total number of characters sent    	|
|     total_char_r|      int        	|     Total number of characters received     	|
|     text_reciprocity_incoming |      int        	|    The total number of times a text is sent to a unique person without response    	|
|     text_reciprocity_outgoing |      int        	|    The total number of times a text is received by a unique person without response       	|

## Further Details

The Willow tree is Forest's simplest data analysis. The code explains itself well and is very
readable. It [can be found
here](https://github.com/onnela-lab/forest/blob/documentation/src/forest/willow/log_stats.py) and is
in a single file.

`log_stats_main` and `log_stats_inner` are near the bottom and handle the business of running
through each analysis participant-by-participant, each analysis function fills out their columns.
