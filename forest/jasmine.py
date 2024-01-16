from forest.jasmine.traj2stats import gps_stats_main, Frequency



def main():
    #study_folder = "/Volumes/gustaf_kata/smith/man_dl/ext"
    #output_folder = "/Volumes/gustaf_kata/smith/output/jasmine"
    #study_folder = "/Volumes/One Touch/cnoc/gps_decrypt"
    study_folder = "/Volumes/One Touch/cnoc/gps_decrypt/"
    output_folder = "/Volumes/One Touch/cnoc/gps_stats"
    tz_str = "America/New_York"
    option = "both"
    time_start = "2018-01-01 00_00_00"
    time_end = "2023-03-01 00_00_00"

    frequency = Frequency.DAILY
    save_traj = False


    gps_stats_main( study_folder,
                    output_folder, 
                    tz_str,
                    frequency, 
                    save_traj, 
                    places_of_interest = None,
                    osm_tags = None, 
                    time_start = None, 
                    time_end = None,
                    participant_ids = None,
                    parameters = None,
                    all_memory_dict = None,
                    all_bv_set = None)




if __name__ == '__main__':
    main()