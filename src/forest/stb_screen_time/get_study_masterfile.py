from datetime import datetime, timedelta
import os

import pandas as pd

from . import config


def excel_numeric_to_date3(val):
    if pd.isna(val) or not str(val).startswith('4'):
        return None
    date_val = int(float(val))
    return datetime(1899, 12, 30) + timedelta(days=date_val)


# File paths
ext_file_path = os.path.join(os.getcwd(), "data_nock_lab", "data_participants_other", "BeiweID_11_9_2022.xlsx")
dat_F_path = os.path.join(os.getcwd(), "results", "beiwe_id_list_with_dates_2022-11-17.csv")
dat_demog_fc_path = os.path.join(os.getcwd(), "data_nock_lab", "data_participants_other", "fc_u01_participant_enrollment_demog.csv")
dat_demog_nofc_path = os.path.join(os.getcwd(), "data_nock_lab", "data_participants_other", "u01_participant_enrollment_demog.csv")
cleaned_demog_path = os.path.join(os.getcwd(), "data_nock_lab", "data_participants_other_processed", "demog_clean.csv")

# Read and clean external data: study start and end
ext_file0 = pd.read_excel(ext_file_path, sheet_name=0).rename(columns=str.lower)

# Remove participants to be excluded
ext_file1 = ext_file0[ext_file0['exclude_p'].isna()]

# Format dates
# ext_file1['study_start'] = ext_file1['study_start'].apply(excel_numeric_to_date3)
ext_file1.loc[:, 'study_start'] = ext_file1['study_start'].apply(excel_numeric_to_date3)

# those are yielding >168 days, cap at 168
# (use variables from the config file -- not publicly available to keep them anonymous)
ext_file1.loc[ext_file1['beiwe_id'] == config.BEIWE_ID_FIX_1, 'study_start'] = pd.to_datetime('2020-09-25')
ext_file1.loc[ext_file1['beiwe_id'] == config.BEIWE_ID_FIX_2, 'study_start'] = pd.to_datetime('2021-11-19')

for fix_id in [config.BEIWE_ID_FIX_3, config.BEIWE_ID_FIX_4, config.BEIWE_ID_FIX_5]:
    if fix_id in ext_file1['beiwe_id'].values:
        start_date = ext_file1.loc[ext_file1['beiwe_id'] == fix_id, 'study_start']
        if start_date.notna().all():
            ext_file1.loc[ext_file1['beiwe_id'] == fix_id, 'study_end'] = start_date + pd.DateOffset(days=168)

# Ensure 'study_start' and 'study_end' are in datetime format
ext_file1.loc[:, 'study_start'] = pd.to_datetime(ext_file1['study_start'], errors='coerce')
ext_file1.loc[:, 'study_end'] = pd.to_datetime(ext_file1['study_end'], errors='coerce')

print(ext_file1[['study_start', 'study_end']].dtypes)

# Calculate the duration of observation
ext_file1['obs_duration'] = (ext_file1['study_end'] - ext_file1['study_start']).dt.days

# Save cleaned data to CSV
ext_file1.to_csv(dat_F_path, index=False)

# Read demographics
dat_demog_fc = pd.read_csv(dat_demog_fc_path).rename(columns=str.lower)
dat_demog_nofc = pd.read_csv(dat_demog_nofc_path).rename(columns=str.lower)

# Process demographics data
dat_demog_fc['age_cat'] = 'adol'
dat_demog_fc = dat_demog_fc.rename(columns={'sex2_new': 'sex', 'race2_new': 'race', 'age_new': 'age'})
dat_demog_fc = dat_demog_fc.dropna(subset=['beiwe_id']).query("beiwe_id != ''")

dat_demog_nofc['age_cat'] = 'adult'
dat_demog_nofc = dat_demog_nofc.rename(columns={'sex_new': 'sex', 'race2_new': 'race', 'age_new': 'age'})
dat_demog_nofc = dat_demog_nofc.dropna(subset=['beiwe_id']).query("beiwe_id != ''")

# Combine demographics data
dat_demog = pd.concat([dat_demog_fc, dat_demog_nofc])

# Remove duplicate beiwe_id
dat_demog = dat_demog.drop_duplicates(subset='beiwe_id', keep='first')

# Save cleaned demographics
dat_demog.to_csv(cleaned_demog_path, index=False)

# Create master file with combined data
ext_file_F = ext_file1.copy()
ext_file_F['has_start_end'] = 1
dat_demog_tojoin = dat_demog.copy()
dat_demog_tojoin['has_demog'] = 1

beiwe_masterfile = ext_file_F.merge(dat_demog_tojoin, on='beiwe_id', how='outer')
beiwe_masterfile['has_start_end'] = beiwe_masterfile['has_start_end'].fillna(0).astype(int)
beiwe_masterfile['has_demog'] = beiwe_masterfile['has_demog'].fillna(0).astype(int)

# Save master file
beiwe_masterfile_path = os.path.join(os.getcwd(), "data_nock_lab", "data_participants_other_processed", "beiwe_id_masterfile.csv")
beiwe_masterfile.to_csv(beiwe_masterfile_path, index=False)
