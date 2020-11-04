# JP Onnela / 12/10/2019
#
# Usage: python beiwe.py --help
# Usage: python beiwe.py --users


import sys
from os import listdir
from datetime import datetime
from datetime import timedelta
from collections import OrderedDict

def get_users(folder):

    data_folder = "./" + folder + "/"
    beiwe_ids = listdir(data_folder)
    beiwe_ids.remove(".DS_Store")
    users = {}
    for beiwe_id in beiwe_ids:
        file_folder = data_folder + beiwe_id + "/identifiers/"
        identifier_file = listdir(file_folder)[0]
        counter = 0

        
     ## in normal case there is only one line in identifier
        for line in open(file_folder + identifier_file):
            if counter == 1:
                line = line.rstrip().split(",")
            counter += 1
            time_str = line[1][0:10]
            os = line[6]
        users[beiwe_id] = (time_str, os)
    return users

def sort_users(users):
    users_sorted_by_time = sorted(users.items(), key=lambda tup: tup[1][0]) ## time_str
    first_user = users_sorted_by_time[0]
    last_user = users_sorted_by_time[-1]
    sorted_users = OrderedDict()
    for (user, value) in users_sorted_by_time:
        sorted_users[user] = value
    return (sorted_users, first_user, last_user)


def list_users(users, first_user, last_user) :
    os_count = {}
    for (key, value) in users.items():
        time_str = value[0]
        os = value[1]        
        print(" ", key, " ", time_str, " ", os)
        if not os in os_count:
            os_count[os] = 1
        else:
            os_count[os] += 1
    
    os_strings = []
    for (key, value) in os_count.items():
        os_strings.append(str(value) + " " + key)
    os_output = os_strings[0] + " and " + os_strings[1]
    first_time_str = first_user[1][0]
    last_time_str = last_user[1][0]
    datetimeFormat = "%Y-%m-%d"
    diff = datetime.strptime(last_time_str, datetimeFormat) - datetime.strptime(first_time_str, datetimeFormat)
    print("\n  Listed %d users enrolled over %d days between %s and %s." % (len(users), diff.days, first_time_str, last_time_str))
    print("  Found a total of %s and %s devices." % (os_strings[0], os_strings[1]))

def users(folder):
    print()
    users = get_users(folder)
    (sorted_users, first_user, last_user) = sort_users(users)
    list_users(sorted_users, first_user, last_user) 
    print()


# -----------------------------------------------------------------------------
## more checks on inputs before executing the function "users"
# check if any arguments provided
if len(sys.argv) == 1:
    print("  Please provide input arguments. Exiting.")
    sys.exit()

elif len(sys.argv) == 2:
    # print help
    if sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print("  Listing functions:")
        print("    --help or -h: list help")
        print("    --users or -u: list Beiwe IDs and enrollment dates")
        print("")

elif len(sys.argv) == 3:
    # list users
    if sys.argv[1] == "--users" or sys.argv[1] == "-u":
        folder = sys.argv[2]
        users(folder)




