import os
import sys
from datetime import datetime
from itertools import islice
from pathlib import Path

import pandas as pd
import requests

import mano

SPACE = "    "
BRANCH = "│   "
TEE = "├── "
LAST = "└── "


def tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
):
    """Given a directory Path object print a visual tree structure"""
    dir_path = Path(dir_path)  # accept string coerceable to Path
    files = 0
    directories = 0

    def inner(directory: Path, prefix: str = "", level2: int = -1):
        nonlocal files, directories
        if not level2:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in directory.iterdir() if d.is_dir()]
        else:
            contents = list(directory.iterdir())
        pointers = [TEE] * (len(contents) - 1) + [LAST]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = BRANCH if pointer == TEE else SPACE
                yield from inner(
                    path, prefix + extension, level2 - 1
                )
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    print(dir_path.name)
    iterator = inner(dir_path, level2=level)
    for line in islice(iterator, length_limit):
        print(line)
    if next(iterator, None):
        print(f"... length_limit, {length_limit}, reached, counted:")
    print(
        f"\n{directories} directories" + (f", {files} files" if files else "")
        )


def concatenate_summaries(dir_path: Path, output_filename: str):
    """Concatenate subject-specific GPS- or communication-related summaries

    Checks to see if there is an hourly or daily folder first,
    then concatenates sub-folders first.
    """
    dir_path = Path(dir_path)  # accept string coercible to Path
    if os.path.exists(dir_path / "hourly"):
        concatenate_folder(
            dir_path / "hourly", output_filename[0:-4] + "_hourly" + ".csv"
        )
    if os.path.exists(dir_path / "daily"):
        concatenate_folder(
            dir_path / "daily", output_filename[0:-4] + "_daily" + ".csv"
        )
    concatenate_folder(dir_path, output_filename)


def concatenate_folder(dir_path: Path, output_filename: str):
    """Concatenate one folder of GPS- or communication-related summaries"""

    # initialize dataframe list
    df_list = []

    # loop through files in dir_path
    for file in os.listdir(dir_path):
        # obtain subject study_id
        file_dir = os.path.join(dir_path, file)
        subject_id = os.path.basename(file_dir)[:-4]
        if file.endswith(".csv"):
            temp_df = pd.read_csv(file_dir)
            temp_df.insert(loc=0, column="Beiwe_ID", value=subject_id)
            df_list.append(temp_df)

    if len(df_list) > 0:

        # concatenate dataframes within list --> Final Data for trajectories
        response_data = pd.concat(df_list, axis=0)

        # make directory
        os.makedirs(dir_path / "concatenated", exist_ok=True)
        path_resp = os.path.join(dir_path / "concatenated", output_filename)

        # write to csv
        response_data.to_csv(path_resp, index=False)
        print(
            "Concatenated folder " + str(dir_path)
            + " to " + str(output_filename)
        )
    else:
        print("No input data found in folder " + str(dir_path))


def download_data(
    keyring,
    study_id,
    download_folder,
    users=None,
    time_start="2008-01-01",
    time_end=None,
    data_streams=None,
):
    """
    Downloads all data for specified users, time frame, and data streams.

    This function downloads all data for selected users,
    time frame, and data streams, and writes them to an
    output folder, with one subfolder for each user, and subfolders
    inside the user's folder for each data stream.
    If a server failure happens, the function re-attempts the download.

    Args:
        keyring: a keyring generated by mano.keyring

        users(iterable): A list of users to download data for.
            If none are entered, it attempts to download data for all users

        study_id(str): The id of a study

        download_folder(str): path to a folder to download data

        time_start(str): The initial date to download data
            (Formatted in YYYY-MM-DD). Default is 2008-01-01, which is
            before any Beiwe data existed.

        time_end(str): The date to end downloads.
            The default is today at midnight.

        data_streams(iterable): A list of all data streams to download.
            The default (None) is all possible data streams.

    """
    if study_id == "":
        print("Error: Study ID is blank")
        return

    if (
        keyring["USERNAME"] == ""
        or keyring["PASSWORD"] == ""
        or keyring["ACCESS_KEY"] == ""
        or keyring["SECRET_KEY"] == ""
    ):
        print("Error: Did you set up the keyring_studies.py file?")
        return

    if not os.path.isdir(download_folder):
        os.makedirs(download_folder, exist_ok=True)

    if time_end is None:
        time_end = datetime.today().strftime("%Y-%m-%d") + "T23:59:00"

    if users is None:
        print("Obtaining list of users...")
        num_tries = 0
        while num_tries < 5:
            try:
                users = [u for u in mano.users(keyring, study_id)]
                break
            except KeyboardInterrupt:
                print("Someone closed the program")
                sys.exit()
            except mano.APIError as e:
                print("Something is wrong with your credentials:")
                print(e)
                break
            except Exception:
                num_tries += 1

    for u in users:
        zf = None
        num_tries = 0
        while num_tries < 5:
            print(f"Downloading data for {u}")
            try:
                zf = mano.sync.download(
                    keyring,
                    study_id,
                    u,
                    data_streams,
                    time_start=time_start,
                    time_end=time_end,
                )
                break
            except requests.exceptions.ChunkedEncodingError:
                print(f"Network failed in download of {u}, try {num_tries}")
                num_tries += 1
            except KeyboardInterrupt:
                print("Someone closed the program")
                sys.exit()
            except mano.APIError as e:
                print("Something is wrong with your credentials:")
                print(e)
                break

        if num_tries == 5:
            print(f"Too many failures; skipping user {u}")
            continue

        if zf is None:
            print(f"No data for {u}; nothing written")
        else:
            zf.extractall(download_folder)
