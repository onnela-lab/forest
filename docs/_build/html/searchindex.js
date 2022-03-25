Search.setIndex({docnames:["index","source/forest","source/forest.bonsai","source/forest.bonsai.tests","source/forest.jasmine","source/forest.jasmine.tests","source/forest.poplar","source/forest.poplar.classes","source/forest.poplar.constants","source/forest.poplar.functions","source/forest.poplar.legacy","source/forest.poplar.raw","source/forest.sycamore","source/forest.willow","source/modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["index.rst","source/forest.rst","source/forest.bonsai.rst","source/forest.bonsai.tests.rst","source/forest.jasmine.rst","source/forest.jasmine.tests.rst","source/forest.poplar.rst","source/forest.poplar.classes.rst","source/forest.poplar.constants.rst","source/forest.poplar.functions.rst","source/forest.poplar.legacy.rst","source/forest.poplar.raw.rst","source/forest.sycamore.rst","source/forest.willow.rst","source/modules.rst"],objects:{"":[[1,0,0,"-","forest"]],"forest.bonsai":[[2,0,0,"-","simulate_gps_data"],[2,0,0,"-","simulate_log_data"],[3,0,0,"-","tests"]],"forest.bonsai.simulate_gps_data":[[2,1,1,"","Action"],[2,1,1,"","ActionType"],[2,1,1,"","Attributes"],[2,1,1,"","Occupation"],[2,1,1,"","Person"],[2,1,1,"","PossibleExits"],[2,1,1,"","Vehicle"],[2,4,1,"","bounding_box"],[2,4,1,"","gen_all_traj"],[2,4,1,"","gen_basic_pause"],[2,4,1,"","gen_basic_traj"],[2,4,1,"","gen_route_traj"],[2,4,1,"","generate_addresses"],[2,4,1,"","generate_nodes"],[2,4,1,"","get_basic_path"],[2,4,1,"","get_path"],[2,4,1,"","gps_to_csv"],[2,4,1,"","load_attributes"],[2,4,1,"","prepare_data"],[2,4,1,"","process_switches"],[2,4,1,"","remove_data"],[2,4,1,"","sim_gps_data"]],"forest.bonsai.simulate_gps_data.Action":[[2,2,1,"","action"],[2,2,1,"","destination_coordinates"],[2,2,1,"","duration"],[2,2,1,"","preferred_exit"]],"forest.bonsai.simulate_gps_data.ActionType":[[2,2,1,"","FLIGHT_PAUSE_FLIGHT"],[2,2,1,"","PAUSE"],[2,2,1,"","PAUSE_NIGHT"]],"forest.bonsai.simulate_gps_data.Occupation":[[2,2,1,"","NONE"],[2,2,1,"","SCHOOL"],[2,2,1,"","WORK"]],"forest.bonsai.simulate_gps_data.Person":[[2,3,1,"","calculate_trip"],[2,3,1,"","choose_action"],[2,3,1,"","choose_preferred_exit"],[2,3,1,"","end_of_day_reset"],[2,3,1,"","set_active_status"],[2,3,1,"","set_travelling_status"],[2,3,1,"","update_preferred_places"]],"forest.bonsai.simulate_gps_data.PossibleExits":[[2,2,1,"","BAR"],[2,2,1,"","CAFE"],[2,2,1,"","CINEMA"],[2,2,1,"","DANCE"],[2,2,1,"","FITNESS"],[2,2,1,"","PARK"],[2,2,1,"","RESTAURANT"]],"forest.bonsai.simulate_gps_data.Vehicle":[[2,2,1,"","BICYCLE"],[2,2,1,"","BUS"],[2,2,1,"","CAR"],[2,2,1,"","FOOT"]],"forest.bonsai.simulate_log_data":[[2,4,1,"","exist_text_call"],[2,4,1,"","gen_call_dur"],[2,4,1,"","gen_call_files"],[2,4,1,"","gen_dir"],[2,4,1,"","gen_random_id"],[2,4,1,"","gen_round"],[2,4,1,"","gen_status"],[2,4,1,"","gen_text_files"],[2,4,1,"","gen_text_len"],[2,4,1,"","gen_timestamp_call"],[2,4,1,"","gen_timestamp_text"],[2,4,1,"","int2str"],[2,4,1,"","number_of_distinct_inds"],[2,4,1,"","sim_log_data"]],"forest.bonsai.tests":[[3,0,0,"-","test_simulate_gps_data"]],"forest.bonsai.tests.test_simulate_gps_data":[[3,4,1,"","coords1"],[3,4,1,"","coords2"],[3,4,1,"","coords3"],[3,4,1,"","directions1"],[3,4,1,"","generated_trajectory"],[3,4,1,"","mock_get_path"],[3,4,1,"","random_path"],[3,4,1,"","sample_addresses"],[3,4,1,"","sample_attributes"],[3,4,1,"","sample_coordinates"],[3,4,1,"","sample_locations"],[3,4,1,"","sample_person"],[3,4,1,"","test_attributes_user_missing_args"],[3,4,1,"","test_bounding_box_simple_case"],[3,4,1,"","test_choose_action_after_work"],[3,4,1,"","test_choose_action_day_home_action"],[3,4,1,"","test_choose_action_day_home_exit"],[3,4,1,"","test_choose_action_day_home_location"],[3,4,1,"","test_choose_action_day_night_action"],[3,4,1,"","test_choose_action_day_night_exit"],[3,4,1,"","test_choose_action_day_night_location"],[3,4,1,"","test_choose_action_office_code"],[3,4,1,"","test_choose_action_office_location"],[3,4,1,"","test_choose_action_simple_case_actions"],[3,4,1,"","test_choose_action_simple_case_times"],[3,4,1,"","test_choose_preferred_exit_morning_home"],[3,4,1,"","test_choose_preferred_exit_night_home"],[3,4,1,"","test_choose_preferred_exit_random_exit"],[3,4,1,"","test_end_of_day_reset"],[3,4,1,"","test_gen_all_traj_consistent_values"],[3,4,1,"","test_gen_all_traj_dist_travelled"],[3,4,1,"","test_gen_all_traj_len"],[3,4,1,"","test_gen_all_traj_time"],[3,4,1,"","test_gen_all_traj_time_at_home"],[3,4,1,"","test_gen_basic_pause_location"],[3,4,1,"","test_gen_basic_pause_t_diff_range"],[3,4,1,"","test_gen_basic_pause_t_e_range"],[3,4,1,"","test_gen_basic_traj_cols"],[3,4,1,"","test_gen_basic_traj_distance"],[3,4,1,"","test_gen_basic_traj_time"],[3,4,1,"","test_gen_route_traj_distance"],[3,4,1,"","test_gen_route_traj_shape"],[3,4,1,"","test_gen_route_traj_time"],[3,4,1,"","test_get_basic_path_length_by_bicycle"],[3,4,1,"","test_get_basic_path_length_by_bus"],[3,4,1,"","test_get_basic_path_length_by_car"],[3,4,1,"","test_get_basic_path_simple_case"],[3,4,1,"","test_get_path_close_locations"],[3,4,1,"","test_get_path_distance"],[3,4,1,"","test_get_path_ending_longitude"],[3,4,1,"","test_get_path_starting_latitude"],[3,4,1,"","test_load_attributes_attributes"],[3,4,1,"","test_load_attributes_nusers"],[3,4,1,"","test_load_attributes_switches"],[3,4,1,"","test_person_cafe_places"],[3,4,1,"","test_person_main_employment"],[3,4,1,"","test_person_office_address"],[3,4,1,"","test_person_office_days"],[3,4,1,"","test_prepare_data_shape"],[3,4,1,"","test_prepare_data_timezones"],[3,4,1,"","test_process_attributes_arguments_correct"],[3,4,1,"","test_process_switches"],[3,4,1,"","test_remove_data_len"],[3,4,1,"","test_set_active_status"],[3,4,1,"","test_set_travelling_status"],[3,4,1,"","test_sim_gps_data_multiple_people"],[3,4,1,"","test_sim_gps_data_times"],[3,4,1,"","test_update_preferred_places_case_first_option"],[3,4,1,"","test_update_preferred_places_case_last_option"],[3,4,1,"","test_zero_meters_bounding_box"]],"forest.jasmine":[[4,0,0,"-","data2mobmat"],[4,0,0,"-","mobmat2traj"],[4,0,0,"-","sogp_gps"],[5,0,0,"-","tests"],[4,0,0,"-","traj2stats"]],"forest.jasmine.data2mobmat":[[4,4,1,"","ExistKnot"],[4,4,1,"","ExtractFlights"],[4,4,1,"","GPS2MobMat"],[4,4,1,"","InferMobMat"],[4,4,1,"","cartesian"],[4,4,1,"","collapse_data"],[4,4,1,"","great_circle_dist"],[4,4,1,"","pairwise_great_circle_dist"],[4,4,1,"","shortest_dist_to_great_circle"],[4,4,1,"","unique"]],"forest.jasmine.mobmat2traj":[[4,4,1,"","I_flight"],[4,4,1,"","Imp2traj"],[4,4,1,"","ImputeGPS"],[4,4,1,"","K1"],[4,4,1,"","adjust_direction"],[4,4,1,"","checkbound"],[4,4,1,"","create_tables"],[4,4,1,"","locate_home"],[4,4,1,"","multiplier"],[4,4,1,"","num_sig_places"]],"forest.jasmine.sogp_gps":[[4,4,1,"","BV_select"],[4,4,1,"","K0"],[4,4,1,"","SOGP"],[4,4,1,"","update_K"],[4,4,1,"","update_Q"],[4,4,1,"","update_alpha"],[4,4,1,"","update_alpha_hat"],[4,4,1,"","update_alpha_vec"],[4,4,1,"","update_c"],[4,4,1,"","update_c_hat"],[4,4,1,"","update_c_mat"],[4,4,1,"","update_e_hat"],[4,4,1,"","update_eta"],[4,4,1,"","update_gamma"],[4,4,1,"","update_k"],[4,4,1,"","update_q"],[4,4,1,"","update_q_mat"],[4,4,1,"","update_s"],[4,4,1,"","update_s_hat"],[4,4,1,"","update_s_mat"]],"forest.jasmine.tests":[[5,0,0,"-","test_traj2stats"]],"forest.jasmine.tests.test_traj2stats":[[5,4,1,"","coords1"],[5,4,1,"","coords2"],[5,4,1,"","sample_nearby_locations"],[5,4,1,"","sample_trajectory"],[5,4,1,"","test_gps_summaries_datetime_nighttime_shape"],[5,4,1,"","test_gps_summaries_log_format"],[5,4,1,"","test_gps_summaries_obs_day_night"],[5,4,1,"","test_gps_summaries_places_of_interest"],[5,4,1,"","test_gps_summaries_shape"],[5,4,1,"","test_transform_point_to_circle_radius"],[5,4,1,"","test_transform_point_to_circle_simple_case"],[5,4,1,"","test_transform_point_to_circle_zero_radius"]],"forest.jasmine.traj2stats":[[4,1,1,"","Frequency"],[4,1,1,"","Hyperparameters"],[4,4,1,"","get_nearby_locations"],[4,4,1,"","gps_quality_check"],[4,4,1,"","gps_stats_main"],[4,4,1,"","gps_summaries"],[4,4,1,"","transform_point_to_circle"]],"forest.jasmine.traj2stats.Frequency":[[4,2,1,"","BOTH"],[4,2,1,"","DAILY"],[4,2,1,"","HOURLY"]],"forest.jasmine.traj2stats.Hyperparameters":[[4,2,1,"","a1"],[4,2,1,"","a2"],[4,2,1,"","accuracylim"],[4,2,1,"","b1"],[4,2,1,"","b2"],[4,2,1,"","b3"],[4,2,1,"","d"],[4,2,1,"","g"],[4,2,1,"","h"],[4,2,1,"","itrvl"],[4,2,1,"","l1"],[4,2,1,"","l2"],[4,2,1,"","l3"],[4,2,1,"","linearity"],[4,2,1,"","method"],[4,2,1,"","num"],[4,2,1,"","r"],[4,2,1,"","sigma2"],[4,2,1,"","switch"],[4,2,1,"","tol"],[4,2,1,"","w"]],"forest.poplar":[[7,0,0,"-","classes"],[8,0,0,"-","constants"],[9,0,0,"-","functions"],[10,0,0,"-","legacy"],[11,0,0,"-","raw"]],"forest.poplar.classes":[[7,0,0,"-","history"],[7,0,0,"-","registry"],[7,0,0,"-","template"],[7,0,0,"-","trackers"]],"forest.poplar.constants":[[8,0,0,"-","misc"],[8,0,0,"-","time"]],"forest.poplar.functions":[[9,0,0,"-","helpers"],[9,0,0,"-","holidays"],[9,0,0,"-","io"],[9,0,0,"-","log"],[9,0,0,"-","time"],[9,0,0,"-","timezone"]],"forest.poplar.functions.helpers":[[9,4,1,"","clean_dataframe"],[9,4,1,"","directory_size"],[9,4,1,"","get_windows"],[9,4,1,"","iqr"],[9,4,1,"","join_lists"],[9,4,1,"","sample_range"],[9,4,1,"","sample_std"],[9,4,1,"","sample_var"],[9,4,1,"","sort_by"]],"forest.poplar.functions.holidays":[[9,4,1,"","is_US_holiday"]],"forest.poplar.functions.io":[[9,4,1,"","read_json"],[9,4,1,"","setup_csv"],[9,4,1,"","setup_directories"],[9,4,1,"","write_json"],[9,4,1,"","write_to_csv"]],"forest.poplar.functions.log":[[9,4,1,"","attributes_to_csv"],[9,4,1,"","log_to_csv"]],"forest.poplar.functions.time":[[9,4,1,"","between_days"],[9,4,1,"","convert_seconds"],[9,4,1,"","local_now"],[9,4,1,"","next_day"],[9,4,1,"","reformat_datetime"],[9,4,1,"","round_timestamp"],[9,4,1,"","to_readable"],[9,4,1,"","to_timestamp"]],"forest.poplar.functions.timezone":[[9,4,1,"","get_offset"],[9,4,1,"","get_timezone"]],"forest.poplar.legacy":[[10,0,0,"-","common_funcs"]],"forest.poplar.legacy.common_funcs":[[10,4,1,"","datetime2stamp"],[10,4,1,"","filename2stamp"],[10,4,1,"","read_data"],[10,4,1,"","stamp2datetime"],[10,4,1,"","write_all_summaries"]],"forest.poplar.raw":[[11,0,0,"-","readers"]],"forest.sycamore":[[12,0,0,"-","common"],[12,0,0,"-","responses"]],"forest.sycamore.common":[[12,4,1,"","aggregate_surveys"],[12,4,1,"","aggregate_surveys_config"],[12,4,1,"","aggregate_surveys_no_config"],[12,4,1,"","convert_timezone_df"],[12,4,1,"","parse_surveys"],[12,4,1,"","q_types_standardize"],[12,4,1,"","read_and_aggregate"],[12,4,1,"","read_json"]],"forest.sycamore.responses":[[12,4,1,"","agg_changed_answers"],[12,4,1,"","agg_changed_answers_summary"],[12,4,1,"","subset_answer_choices"]],"forest.willow":[[13,0,0,"-","log_stats"]],"forest.willow.log_stats":[[13,4,1,"","comm_logs_summaries"],[13,4,1,"","log_stats_main"]],forest:[[2,0,0,"-","bonsai"],[1,0,0,"-","constants"],[4,0,0,"-","jasmine"],[6,0,0,"-","poplar"],[13,0,0,"-","willow"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function"},terms:{"0":[2,4,9,10],"002":4,"01":4,"05":4,"1":[2,4,9],"10":[2,4,9],"100":4,"1000":9,"16":5,"180":4,"1d":4,"2":[2,4,9],"200":4,"23":10,"250":3,"2592000":4,"2d":[2,4],"3":[2,4,9],"4":[2,4,9],"5":4,"50":9,"51":4,"51070891":9,"60":9,"60000":9,"7":4,"8":[4,9],"864000":4,"boolean":2,"case":3,"class":[1,2,4,6],"default":[4,12],"do":[4,9],"enum":[2,4],"export":9,"final":[2,4,12],"float":[2,4,9],"function":[1,2,3,4,6,11,12],"import":10,"int":[2,4,9,10],"new":[2,4,9],"return":[2,4,9,10,12,13],"switch":[2,3,4],"true":[2,4,9,12],"try":9,A:[2,9,12],For:[9,10],If:[4,9,12],In:4,It:4,The:[4,9,10,12,13],There:9,These:[],To:2,Will:9,a1:4,a2:4,acceleromet:10,access:[],accord:9,accuraci:4,accuracylim:4,action:[2,3],actiontyp:2,activ:2,active_statu:2,actual:3,add:[9,12],addition:4,address:2,adjust:4,adjust_direct:4,after:[2,4,9,10],afternoon:3,agg:12,agg_changed_answ:12,agg_changed_answers_summari:12,aggreg:12,aggregate_survei:12,aggregate_surveys_config:12,aggregate_surveys_no_config:12,algorithm:[4,7],all:[2,3,4,5,9,10,12],all_bv_set:4,all_data:12,all_memory_dict:4,all_timezon:10,along:2,alpha:4,alreadi:4,also:[4,9],altitud:[],amen:[2,4],amount:5,an:[2,4,7,9,12],android:12,ani:[2,4,9,12],answer:12,answers_l:12,anticip:7,anylist:4,anywher:2,api:[1,2,4],api_kei:[2,3],app_log:[],appear:4,applic:4,approxim:5,ar:[2,3,4,9,10,12,13],arbitrari:[4,9],area:2,arg:[2,4,9,10,13],argument:[3,9],around:[2,3],arrai:[2,4,10],arrang:9,asctim:9,attribut:[2,3,9],attribute_list:9,attributes_dict:2,attributes_to_csv:9,audio_record:[],av_flight_dur:4,av_flight_length:4,avail:[4,9,10,12],available_attribut:9,averag:[4,12],avgmat:4,avoid:4,aw:9,axis:4,b1:4,b2:4,b3:4,b:[4,9],back:2,bar:2,base:[1,2,4,14],basi:4,basic:[2,3],basicconfig:9,been:2,befor:[4,9,10],begin:[9,12],behavior:3,beiw:[4,8,9,10,11,12,13],beiwe_id:[12,13],below:4,better:[4,9],between:[2,4,9,12],between_dai:9,bi:4,bicycl:2,bigger:4,binari:4,bluetooth:[],bonsai:[1,14],bool:[2,4,9,12],both:[2,4,9],bound:[2,3],boundari:4,bounding_box:2,box:[2,3],bu:2,bug:4,bundl:[],button:[],bv:4,bv_index:4,bv_select:4,bv_set:4,c:[4,9],cafe:[2,3],cafe_plac:3,calcul:[2,4,7,9],calculate_trip:2,call:[1,4,10],can:[9,10],car:2,cartesian:4,categor:7,categori:3,center:[2,4,5],centr:4,chang:[2,3,4,9,12],check:[4,5,9,10,12],checkbound:4,checkbox:[],choos:3,choose_act:2,choose_preferred_exit:2,chunk:4,cinema:2,circl:[4,5],citi:2,city_nam:2,clean:[9,12],clean_datafram:9,clock:9,close:4,closest:9,code:[2,4,9],coeffici:4,col:4,collapse_data:4,collect:2,columm:4,column:[2,3,4,9,12],com:9,combin:4,comm_logs_summari:13,common:[1,9,14],common_func:[1,6],commun:10,compar:4,complet:[2,4],concord:[],conduct:[10,13],config:12,config_path:12,configur:[9,12],consecut:4,consid:4,consist:3,constant:[6,14],contain:[1,2,4,5,7,9,10],content:14,continu:4,control:4,conveni:[],convent:[],convert:[9,10,12],convert_second:9,convert_timezone_df:12,cooordin:4,coordiant:4,coordin:[2,4,5,9],coords1:[3,5],coords2:[3,5],coords3:3,copi:9,core:[2,4,12],correct:[2,3,5,12],correctli:3,correspond:[4,9,10],could:4,count:4,countri:2,country_2_letter_iso_cod:2,cover:9,creat:[2,4,5,9,12],create_t:4,criterion:4,csv:[2,4,9,10],cumul:4,current:[2,4,9],current_i:4,current_t:4,current_tim:2,current_x:4,cycl:2,d:[4,9],dai:[2,3,4,5,9,10,12],daili:[4,13],danc:2,data2mobmat:[1,14],data:[2,3,4,7,8,9,10,11,12],data_fold:[],data_stream:12,dataclass:2,datafram:[2,4,9,10,12,13],dataset:[3,4,12],datastream:[10,12],date:[2,5,9,12],date_format:9,date_list:9,datetim:[2,4,5,9],datetime2stamp:10,datetime_str:9,day_of_week:2,daytim:4,debug:[9,13],decid:2,defin:4,degre:9,delta_i:4,delta_x:4,depend:2,deriv:4,deseri:9,dest_i:4,dest_t:4,dest_x:4,destin:[2,4],destination_coordin:2,detail:12,determiend:4,determin:[4,13],dev:2,devic:[7,12],device_o:[],devicemot:[],df:[9,12],df_call:13,df_merg:12,df_text:13,diamet:4,dict:[2,4,9,12],dictionari:[2,4,9,12],differ:[3,4],difficulti:4,direct:[4,9],directions1:3,directli:9,directori:[7,9],directory_s:9,dirpath:9,dirpath_list:9,discov:4,dispac:4,dist:4,dist_travel:4,distanc:[2,3,4],divid:4,doc:[1,6,9],docstr:[10,13],document:[],doe:[2,4],don:4,download:12,drop:9,drop_dupl:9,duplic:12,dur:2,durat:[2,4],dure:[4,9],e:[7,9],e_hat:4,each:[2,4,9,12],earli:3,earliest:10,effect:4,element:[2,4],employ:2,empti:[4,7],enabl:7,encod:[],end:[2,3,4,9,10,12],end_dat:[2,9],end_i:4,end_of_day_reset:2,end_x:4,entri:9,entropi:4,enumer:[2,4],equal:4,error:2,eta:4,etc:[9,12],evenli:9,event:[],everi:4,exampl:9,exce:4,exist:9,exist_text_cal:2,existknot:4,exit:[2,3],exit_cod:2,expect:2,express:9,extended_format:9,extens:9,extra:9,extract:9,extractflight:4,f:9,fail:[2,4],fall:9,fals:[4,9,12],featur:4,field:12,figur:9,file:[2,4,9,10,12],file_count:9,filenam:10,filename2stamp:10,filepath:9,filter:4,find:9,first:[2,3,4,9,12],fit:2,fitness_centr:2,flight:[2,4],flight_pause_flight:2,flight_tim:4,fly:10,folder:[4,10,12],follow:9,fomat:9,foot:2,form:4,format:[2,4,8,9,10],found:9,fourth:4,fpf:2,fraction:4,frame:[2,4,9,12],free_respons:[],frequenc:4,from:[2,3,4,5,9,10,12,13],from_format:9,from_tz:9,full_data:2,funcnam:9,furthurmor:4,futur:4,g:[4,7,9],gamma:4,gaussian:4,gen_all_traj:2,gen_basic_paus:2,gen_basic_traj:2,gen_call_dur:2,gen_call_fil:2,gen_dir:2,gen_random_id:2,gen_round:2,gen_route_traj:2,gen_statu:2,gen_text_fil:2,gen_text_len:2,gen_timestamp_cal:2,gen_timestamp_text:2,gener:[2,3,4,9],generate_address:2,generate_nod:2,generated_trajectori:3,geocent:4,geometri:4,get:[3,9,12],get_basic_path:2,get_nearby_loc:4,get_offset:9,get_path:[2,3],get_timezon:9,get_window:9,given:[3,4,9,10],gl:4,glc:4,go:3,gp:[2,4,9,10,11],gps2mobmat:4,gps_data:2,gps_quality_check:4,gps_stats_main:4,gps_summari:4,gps_to_csv:2,great:4,great_circle_dist:4,greater:4,gtraj_random:[],gtraj_with_one_visit:[],gtraj_with_regular_visit:[],gyro:[],h:[2,4,9],ha:[2,12],handl:7,handler:9,have:[2,4,9,12],header:[4,9],helper:[1,6],here:13,hi:2,higher:4,highlight:9,him:4,histori:[1,6],hold:2,holidai:[1,6],home:[2,3,4],home_coordin:2,home_i:4,home_tim:4,home_time_list:2,home_x:4,hour:[2,4,9,10],hour_m:9,hourli:[4,13],hous:2,house_address:2,how:[9,12],hstack:4,html:9,http:[2,9],human:9,hyperparamet:4,hyperparemet:4,i:[4,9],i_flight:4,id:[4,10,12,13],identifi:[7,9,10],imp2traj:4,imp_:4,imp_t0:4,imp_t1:4,imp_tabl:4,imp_x0:4,imp_x1:4,imp_y0:4,imp_y1:4,implement:7,imput:4,impute2second:[],imputegp:4,includ:[2,9,12],inclus:9,incom:4,increas:3,indent:9,index:[0,4,9,12],indic:[2,4,9],individu:12,infer:4,infermobmat:4,info:[],info_text_box:[],inform:[9,12],input:[4,9,13],instanc:2,instead:10,instruct:9,int2str:2,integ:[4,9,10],intend:[],interest:[4,10],interg:4,intermedi:9,interv:4,intsead:4,io:[1,6,12],iqr:9,is_holidai:9,is_us_holidai:9,item:[4,9],itrvl:4,jasmin:[1,14],join:[4,9],join_list:9,joined_list:9,josh:12,json:[2,4,5,9,12],just:4,k0:4,k1:4,k:[2,4],k_mat:4,keep:2,kei:[2,4,9],keyword:4,knot:4,knownledg:4,kwarg:2,l1:4,l2:4,l3:4,l_:[],l_e:[],label:9,larg:[4,10],largest:4,last:[3,4,9,12],last_answ:12,lat1:4,lat2:4,lat:[2,4],lat_end:4,lat_start:4,late:3,latitud:[3,4,9],latlon_arrai:4,lattitud:2,legaci:[1,6],leisur:4,len:[4,9],length:[3,4,9],less:[2,3,4],level:9,levelnam:9,librari:[9,12],like:[4,9],limit:2,line:[9,12],linear:4,list:[2,4,9,10,12],list_of_list:9,list_to_sort:9,list_to_sort_bi:9,lkp:12,load:[2,3],load_attribut:2,loc:4,loc_i:4,loc_x:4,local:[9,12],local_now:9,local_plac:2,locat:[2,3,4,9],locate_hom:4,location1:[2,4],location2:[2,4],location3:4,location_end:2,location_start:2,log:[1,4,5,6,10],log_dir:9,log_format:9,log_nam:9,log_stat:[1,14],log_stats_main:13,log_to_csv:9,logger:4,logrecord:9,lon1:4,lon2:4,lon:[2,4],lon_end:4,lon_start:4,longer:4,longitud:[2,3,4,9],longitudin:7,longtitud:4,lookup:12,low:4,m:9,made:2,magnetomet:[],mai:10,main:4,main_employ:2,manag:7,mani:[2,4],manner:4,map:12,mark:4,mat:4,match:4,matrix:4,max_dist_hom:4,maximam:4,maximum:[2,4],mean:[2,4],measur:4,meet:4,megabyt:9,memori:[4,9,10],memory_dict:4,merg:12,messag:9,meter:[2,3,4],method:4,metric:4,mi:4,millisecond:9,min:10,min_m:9,minimum:2,minut:[2,4,5,9,10],misc:[1,6],miscellan:8,miss:[2,3,4],missing_str:9,mobmat2traj:[1,14],mobmat:4,mock:3,mock_get_path:3,mocker:[3,5],modif:9,modul:[0,14],month:[4,10],more:4,morn:3,most:12,move:4,movemo:4,msec:9,multipl:[2,4,12],multipli:4,must:[9,12],n:[3,4],n_person:2,name:[4,9,10,12],nan:9,ndarrai:[2,4],ndigit:9,nearbi:4,nearest:9,necessari:9,need:[2,4,7,10,13],next:[4,9],next_dai:9,next_dat:9,night:[2,3,5],nighttim:[4,5],node:[2,3],none:[2,4,9,10,12,13],normal:3,note:9,now:9,np:[2,4,9],num:4,num_sig_plac:4,num_xi:4,number:[2,4,9,12],number_of_distinct_ind:2,numer:[],numpi:[2,4,10],ob:[2,4],object:[2,4,7,9],obs_dai:4,obs_data:2,obs_dur:4,obs_night:4,observ:[2,3,4,5,9],occup:2,occur:10,off_cycl:2,off_period:2,offic:[2,3],office_address:3,offset:9,on_period:2,one:[3,4,10,12],onli:[2,4,13],onlin:[4,7],open:2,openrout:2,openrouteservic:2,openstreetmap:4,optimis:2,option:[2,4,9,12,13],order:[4,9],ordereddict:9,org:[2,9],origim:4,origin:2,origin_i:4,origin_x:4,osm:4,other:[2,4,13],out:[4,9,10,12],output:[4,9,10,12,13],output_fold:[2,4,10,13],output_path:10,over:[4,9,12],overpass:[2,4],overwrit:9,p:[2,4],p_night:2,packag:14,page:0,pair:[4,9],pairwise_great_circle_dist:4,panda:[2,4,9,10,12,13],paper:4,par:4,param:[2,12],paramet:[1,2,4,9,12],park:2,parse_survei:12,participant_id:4,particular:[9,12],path:[2,3,4,9,10,12],path_coordin:2,pattern:4,paus:[2,3,4],pause_night:2,pause_tim:4,pd:[4,10],peic:4,peopl:2,per:12,percentag:[2,4],perform:2,period:[2,4,9],perpendicular:4,persist:7,person:[2,3,4,9],person_point_radiu:4,pickl:4,place:[3,4,5],place_point_radiu:4,placehold:7,places_of_interest:4,platform:[],pleas:10,plu:5,point:[2,4,5],polygon:4,poplar:[1,14],possibl:[2,4],possibleexit:2,potenti:2,power:7,power_ev:[],power_st:[],precis:9,prefer:[2,3],preferr:2,preferred_exit:2,preferred_plac:2,prepar:[2,3],prepare_data:2,preprocess:2,present:10,pretti:9,previou:[4,9],print:9,prob:4,probabl:9,process:[3,4,9,10,12],process_switch:2,produc:2,proport:2,provid:[2,4,10],proxim:[],py:4,python:9,pytz:[9,10,12],q:[4,12],q_types_standard:12,qualiti:4,quality_threshold:4,queri:[2,4],question:[9,12],question_typ:[],question_type_nam:[],quota:2,r:[2,4],radio:[],radio_button:[],radiu:[2,4,5],rais:[2,4],random:[3,4,5],random_path:3,rang:[3,4,9],rate:2,raw:[1,4,6,8,9],reachabl:[],read:[4,9,10,12],read_and_aggreg:12,read_comm_log:13,read_data:[4,10],read_json:[9,12],readabl:9,reader:[1,6],realist:2,reason:10,recommend:4,record:[4,9],redund:12,reformat:9,reformat_datetim:9,registr:10,registri:[1,6],remain:2,remov:[3,12],remove_data:2,replac:9,report:[],repres:[2,4],requir:[4,9],reset:[2,3],resolut:[4,13],respons:[1,14],restaur:2,result:[2,4],round:9,round_timestamp:9,rout:[2,3],row:[4,9],run:[2,4,12],runtimeerror:[2,4],s3:9,s:[2,4,7,9,10,12],s_hat:4,s_mat:4,same:[4,5,9],sampl:[2,3],sample_address:3,sample_attribut:3,sample_coordin:3,sample_loc:3,sample_nearby_loc:5,sample_person:3,sample_rang:9,sample_std:9,sample_trajectori:5,sample_var:9,save:[2,4,10],save_log:4,save_traj:4,scalar:4,scenario:4,schedul:9,school:2,script:4,sd_flight_dur:4,sd_flight_length:4,search:0,sec:10,second:[2,3,4,9,10],see:9,select:[2,4],sensor:9,separ:4,server:12,servic:2,set:[2,4,9],set_active_statu:2,set_travelling_statu:2,setup_csv:9,setup_directori:9,shape:[3,4,5],share:4,shortest:4,shortest_dist_to_great_circl:4,should:[2,4,9,10,12],show:4,sigma2:4,sigmax:4,signific:4,sim_gps_data:2,sim_log_data:2,simialr:4,similar:4,simpl:3,simul:[2,3],simulate_gps_data:[1,3,14],simulate_log_data:[1,14],singl:[2,9,12],size:[4,9],slider:[],slightli:4,small:4,smi:4,smooth:4,so:12,sogp:4,sogp_gp:[1,14],some:9,someth:9,sort:9,sort_bi:9,sorted_list:9,space:9,spars:4,specif:2,specifi:[4,10],spent:[2,4,5,12],split:4,split_day_night:4,stack:[4,12],stackoverflow:9,stamp2datetim:10,stamp:10,stamp_end:[4,13],stamp_start:[4,13],standard:12,start:[2,3,4,9,10],start_dat:[2,9],start_i:4,start_x:4,stat:[4,5,10,13],state:7,statist:[4,5,7],stats_pdfram:10,statu:[2,4],std:9,step:4,str:[2,4,9,10,12,13],stream:[2,12],string:[2,4,9,12],studi:[4,9,10,12,13],study_dir:12,study_fold:[4,10,13],study_id:4,study_tz:12,subdirectori:9,subfold:12,submiss:12,submit:[1,14],submodul:[6,14],subpackag:14,subset:[2,4],subset_answer_choic:12,sum:2,summar:4,summari:[4,5,7,10,12,13],survei:[9,12],survey_answ:[],survey_data:12,survey_tim:[],sycamor:[1,14],system:9,t0:4,t1:4,t:4,t_:[],t_diff:4,t_diff_rang:[2,3],t_e_rang:[2,3],t_xy:4,tag:4,take:[2,4,12],task:9,templat:[1,6],tend:4,termin:9,test:[1,2,4],test_attributes_user_missing_arg:3,test_bounding_box_simple_cas:3,test_choose_action_after_work:3,test_choose_action_day_home_act:3,test_choose_action_day_home_exit:3,test_choose_action_day_home_loc:3,test_choose_action_day_night_act:3,test_choose_action_day_night_exit:3,test_choose_action_day_night_loc:3,test_choose_action_office_cod:3,test_choose_action_office_loc:3,test_choose_action_simple_case_act:3,test_choose_action_simple_case_tim:3,test_choose_preferred_exit_morning_hom:3,test_choose_preferred_exit_night_hom:3,test_choose_preferred_exit_random_exit:3,test_end_of_day_reset:3,test_gen_all_traj_consistent_valu:3,test_gen_all_traj_dist_travel:3,test_gen_all_traj_len:3,test_gen_all_traj_tim:3,test_gen_all_traj_time_at_hom:3,test_gen_basic_pause_loc:3,test_gen_basic_pause_t_diff_rang:3,test_gen_basic_pause_t_e_rang:3,test_gen_basic_traj_col:3,test_gen_basic_traj_dist:3,test_gen_basic_traj_tim:3,test_gen_route_traj_dist:3,test_gen_route_traj_shap:3,test_gen_route_traj_tim:3,test_get_basic_path_length_by_bicycl:3,test_get_basic_path_length_by_bu:3,test_get_basic_path_length_by_car:3,test_get_basic_path_simple_cas:3,test_get_path_close_loc:3,test_get_path_dist:3,test_get_path_ending_longitud:3,test_get_path_starting_latitud:3,test_gps_summaries_datetime_nighttime_shap:5,test_gps_summaries_log_format:5,test_gps_summaries_obs_day_night:5,test_gps_summaries_places_of_interest:5,test_gps_summaries_shap:5,test_load_attributes_attribut:3,test_load_attributes_nus:3,test_load_attributes_switch:3,test_person_cafe_plac:3,test_person_main_employ:3,test_person_office_address:3,test_person_office_dai:3,test_prepare_data_shap:3,test_prepare_data_timezon:3,test_process_attributes_arguments_correct:3,test_process_switch:3,test_remove_data_len:3,test_set_active_statu:3,test_set_travelling_statu:3,test_sim_gps_data_multiple_peopl:3,test_sim_gps_data_tim:3,test_simulate_gps_data:[1,2],test_traj2stat:[1,4],test_transform_point_to_circle_radiu:5,test_transform_point_to_circle_simple_cas:5,test_transform_point_to_circle_zero_radiu:5,test_update_preferred_places_case_first_opt:3,test_update_preferred_places_case_last_opt:3,test_zero_meters_bounding_box:3,text:[9,10,12],than:[2,3,4],thei:[4,12],them:[4,9,12],thi:[2,4,7,9,12],third:[2,4],those:4,threshold:4,through:2,tidi:4,time:[1,2,3,4,5,6,10,12],time_end:[4,10,13],time_list:10,time_start:[2,4,10,13],timestamp:[2,3,4,9,10],timestamp_:2,timezon:[1,2,3,4,6,7,10,12,13],tl:4,to_format:9,to_read:9,to_timestamp:9,to_tz:9,todo:12,togeth:12,tol:4,too:[2,10,12],tool:[7,9],top:4,total:[2,5,9],total_d_list:2,toward:4,track:7,tracker:[1,6],traj2stat:[1,5,14],traj:[2,4],trajectori:[2,3,4,5],transform:4,transform_point_to_circl:4,transport:[2,3],travel:[2,3],travelling_statu:2,travers:2,treat:4,tri:2,trip:2,triplet:4,try_closest:9,ts:9,tupl:[2,4,9,12],two:4,type:[2,4,9,12],tz:[9,10],tz_str:[2,4,10,12,13],tzfile:9,under:[],understand:12,uniqu:4,unit:[2,4,9],univers:2,unix:10,unknown:4,until:10,up:[4,9],updat:[2,4],update_:4,update_alpha:4,update_alpha_hat:4,update_alpha_vec:4,update_c:4,update_c_hat:4,update_c_mat:4,update_e_hat:4,update_eta:4,update_gamma:4,update_index:9,update_k:4,update_preferred_plac:2,update_q:4,update_q_mat:4,update_s_hat:4,update_s_mat:4,us:[1,2,4,9,10,12],user:[2,3,4,10,12],usual:9,utc:[9,10,12],utc_col:12,valid:[4,5],valu:[2,3,4,9,12],valueerror:[2,4],variabl:[4,7],variable_nam:[],vector:4,vehicl:2,visit:4,volum:10,w:4,wa:[2,10,13],wai:9,wait:10,want:[2,4,10],warn:4,watch:4,we:[2,4,7,10,12],week:[2,12],welford:7,were:9,what:[10,12],when:[2,4,9,12],where:[4,9,10,13],whether:4,which:[2,4,9,10],whose:2,wifi:[],willow:[1,14],window:[4,9,10],window_length_m:9,within:[2,4,9],without:[4,9],word:2,work:[2,3,8,9,11],world:2,write:[2,4,9,10],write_all_summari:10,write_json:9,write_to_csv:9,written:9,x0:4,x1:4,x2:4,x:4,x_current:4,y0:4,y1:4,y:[4,9],y_current:4,ye:4,year:[4,10],you:[4,10],z:[],zero:5,zone:12},titles:["Welcome to forest\u2019s documentation!","forest package","forest.bonsai package","forest.bonsai.tests package","forest.jasmine package","forest.jasmine.tests package","forest.poplar package","forest.poplar.classes package","forest.poplar.constants package","forest.poplar.functions package","forest.poplar.legacy package","forest.poplar.raw package","forest.sycamore package","forest.willow package","forest"],titleterms:{"class":7,"function":9,base:12,bonsai:[2,3],common:12,common_func:10,constant:[1,8],content:[1,2,3,4,5,6,7,8,9,10,11,12,13],data2mobmat:4,doc:11,document:0,forest:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],helper:9,histori:7,holidai:9,indic:0,io:9,jasmin:[4,5],legaci:10,log:9,log_stat:13,misc:8,mobmat2traj:4,modul:[1,2,3,4,5,6,7,8,9,10,11,12,13],packag:[1,2,3,4,5,6,7,8,9,10,11,12,13],poplar:[6,7,8,9,10,11],raw:11,reader:11,registri:7,respons:12,s:0,simulate_gps_data:2,simulate_log_data:2,sogp_gp:4,submit:12,submodul:[1,2,3,4,5,7,8,9,10,11,12,13],subpackag:[1,2,4,6],sycamor:12,tabl:0,templat:7,test:[3,5],test_simulate_gps_data:3,test_traj2stat:5,time:[8,9],timezon:9,tracker:7,traj2stat:4,welcom:0,willow:13}})