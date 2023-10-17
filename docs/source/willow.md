# Willow

## Usage
Create summaries of calls and texts data from the Beiwe app. The master function is `willow.log_stats.log_stats_main(study_folder,output_folder,tz_str,option,time_start,time_end,beiwe_id)` and it will return a csv file for each subject in the study folder, with the ID as the filename and each summary statistic about communication logs as a column. The statistics are summarized in a daily/hourly manner based on your choice.  

- `study_folder`, string, the path of the study folder. The study folder should contain individual participant folders with subfolders `calls`,`texts` inside
- `output_folder`, string, the path of the folder where you want to save results
- `tz_str`, string, the timezone where the study is/was conducted. Please use "`pytz.all_timezones`" to check all options. For example, "America/New_York".  
- `option`, 'daily' or 'hourly' or 'both' for the temporal resolution for summary statistics.  
- `time_start`, `time_end` are starting time and ending time of the window of interest.  
     The time should be a list of integers with format [year, month, day, hour, minute, second] (default: None).    
     If `time_start` is None and `time_end` is None: then it reads all the available files.   
     If `time_start` is None and `time_end` is given, then it reads all the files before the given time.   
     If `time_start` is given and `time_end` is None, then it reads all the files after the given time.   
- `beiwe_id`: a list of beiwe IDs. If it is set to None (default), then it is a list of all available beiwe IDs in your study folder.
  

## Installation Instruction: 
`from forest import willow`

___
## Functions

## `willow.log_stats`  

* **`willow.log_stats.log_stats_main`**: Generates csv of summary statistics (see table below) for each Beiwe user (Android phones only).

## Summary statistics

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


## References  

## Contact information for questions: 
[Email the Onnela Lab](mailto:onnela.lab@gmail.com)
