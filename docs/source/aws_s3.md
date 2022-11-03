*Created by*: Marta Karas ([martakarass](https://github.com/martakarass)) on 2022-11-01 

*Credits*: Eli Jones ([biblicabeebli](https://github.com/biblicabeebli)) -- contributed util functions code, created credentials used in testing this code, guided on using s3 approach presented below. 

# Accessing and decrypting Beiwe data stored on Amazon S3

This page provides a guide to read in and decrypt Beiwe data, stored on Amazon S3, with Python code from one's local computer or a remote server. 

The workflow showed below leads to reading in the data as a `pandas` `DataFrame`. These data can be used then for further processing in Python or be saved into one's machine as are; the latter is an example of using the workflow as a way of downloading Beiwe data.  

## Amazon S3

Amazon Web Services (AWS) is one of many cloud server providers. AWS offer various services,
including Amazon Simple Storage Service ([Amazon S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)). 

Amazon S3 is a service to store and protect data for variety of use cases, such as data lakes, cloud-native applications, and mobile apps. Amazon S3 offers a range of storage classes to choose from to optimize the cost (based on preferred features of the data access, resiliency, etc.), allows for controlling access to data, offers replication features to back up and restore critical data, and others.

Beiwe data is stored on Amazon S3 (decrypted at all times).  

## Boto3 

[Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) is a Python software development kit (SDK) for AWS. It allows to access, create, update and delete AWS resources, including Amazon S3, from one's Python script. The workflow showed below makes use of Boto3's functions [GetObject](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.get_object) and [ListObjectsV2](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.list_objects_v2).


## Credentials needed 

Two sets of credentials are needed to follow the presented workflow to read in and decrypt Beiwe data  stored on Amazon S3: 

1. Beiwe data AWS access credentials: AWS access key id, AWS secret access key;
2. Beiwe data decryption credentials: decryption key. 

At the time of writing, these credentials are administered by Eli Jones ([biblicabeebli](https://github.com/biblicabeebli)). 

In addition, we also need to know Amazon S3 region name and Amazon S3 bucket name for the Beiwe data we want to access (both typically to be provided by the administrator), and the study folder name -- equivalent to the Beiwe study ID.

## Workflow setup  

The workflow setup I use builds upon two Python files: one that stores the credential values, and another that stores util functions for reading and decrypting. Both files are given below. 

### Python file with credential values 

The file defines Python environmental variables storing the credential values I received from the administrator. These values are confidential and should only ever be shared via a secure point of access, for example [Harvard Secure File Transfer](https://security.harvard.edu/secure-file-transfer). 

I name the file `aws_read_and_decrypt_creds.py`. The file I use is showed below. Note the true values were replaced by "foobar" placeholders for public presentation. The message printing part is optional.

```python
import os

os.environ['BEIWE_SERVER_AWS_ACCESS_KEY_ID'] = "foobar1"
os.environ['BEIWE_SERVER_AWS_SECRET_ACCESS_KEY'] = "foobar2"
os.environ['S3_REGION_NAME'] = "foobar3"
os.environ['S3_BUCKET'] = "foobar4"
os.environ['STUDY_FOLDER'] = "foobar5"
os.environ['DECRYPTION_KEY_STRING'] = "foobar6"

msg_test = 'Successfully defined the following environmental variables: '\
'BEIWE_SERVER_AWS_ACCESS_KEY_ID, '\
'BEIWE_SERVER_AWS_SECRET_ACCESS_KEY, '\
'S3_REGION_NAME, '\
'S3_BUCKET, '\
'STUDY_FOLDER, '\
'DECRYPTION_KEY_STRING'
print(msg_test)
```

### Python file with util functions 

The file defines `boto3` connection client to to access Amazon S3 services and defines a set of Python util functions to accomplish a task of reading in and decrypting Beiwe data stored on Amazon S3. It assumes environmental variables from `aws_read_and_decrypt_creds.py` have been already defined. 

I name the file `aws_read_and_decrypt_utils.py`. The file I use is showed below. 

```python
import os
import base64
# pip install boto3
import boto3
# pip install pycryptodomex==3.14.1
from Cryptodome.Cipher import AES
from io import StringIO
from io import BytesIO


# define variables based on variables located in the environment 
BEIWE_SERVER_AWS_ACCESS_KEY_ID = os.environ['BEIWE_SERVER_AWS_ACCESS_KEY_ID']
BEIWE_SERVER_AWS_SECRET_ACCESS_KEY = os.environ['BEIWE_SERVER_AWS_SECRET_ACCESS_KEY'] 
S3_REGION_NAME = os.environ['S3_REGION_NAME'] 
S3_BUCKET = os.environ['S3_BUCKET'] 
STUDY_FOLDER = os.environ['STUDY_FOLDER'] 
DECRYPTION_KEY_STRING = os.environ['DECRYPTION_KEY_STRING'] 
DECRYPTION_KEY = DECRYPTION_KEY_STRING.encode()


connection = boto3.client(
    's3',
    aws_access_key_id = BEIWE_SERVER_AWS_ACCESS_KEY_ID,
    aws_secret_access_key = BEIWE_SERVER_AWS_SECRET_ACCESS_KEY,
    region_name = S3_REGION_NAME,
)


def list_processed_files(prefix):
    # sanitize by removing starting and trailing slashes, then test for appropriate prefixes and
    # automatically add them if missing.
    prefix = prefix.strip("/")

    if not prefix:
        prefix = f"CHUNKED_DATA/{STUDY_FOLDER}"

    if not prefix.startswith(f"CHUNKED_DATA/{STUDY_FOLDER}"):
        prefix = f"CHUNKED_DATA/{STUDY_FOLDER}/" + prefix

    paginator = connection.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(
        Bucket = S3_BUCKET, Prefix = prefix)
    for page in page_iterator:
        if 'Contents' not in page:
            return
        for item in page['Contents']:
            yield item['Key']


def list_raw_files(prefix):
    # sanitize by removing starting and trailing slashes, then test for appropriate prefixes and
    # automatically add them if missing.
    prefix = prefix.strip("/")

    if not prefix:
        prefix = f"{STUDY_FOLDER}"

    if not prefix.startswith(f"{STUDY_FOLDER}"):
        prefix = f"{STUDY_FOLDER}/" + prefix

    paginator = connection.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket = S3_BUCKET, Prefix = prefix)
    for page in page_iterator:
        if 'Contents' not in page:
            return
        for item in page['Contents']:
            yield item['Key']


def decrypt_server(data: bytes) -> bytes:
    """ Decrypts file encrypted by Beiwe for S3. Passing in non-bytes, non-memview values into data
    will cause a TypeError. """
    # the iv is the first 16 bytes, followed by the data
    # technically this is a memcopy operation, may be able to optimize by making a memview slice?
    iv = data[:16]
    data = data[16:]
    return AES.new(DECRYPTION_KEY, AES.MODE_CFB, segment_size=8, IV=iv).decrypt(data)


def download(file_path: str) -> bytes:
    raw = connection.get_object(Bucket = S3_BUCKET, Key = file_path,
                                ResponseContentType = 'string')['Body'].read()
    return decrypt_server(raw)
```


## Use example: browsing and reading in Beiwe data 

Import statements:

```python
import sys
import os
import base64
# pip install boto3
import boto3
import pandas as pd
# pip install pycryptodomex==3.14.1
from Cryptodome.Cipher import AES
from io import BytesIO
```

Read the credentials file and the util functions file:  

```python
# add path to the directory where the following files are stored: 
# - aws_read_and_decrypt_creds.py
# - aws_read_and_decrypt_utils.py
sys.path.insert(0, "~/Documents/s3_test")

# read credentials variables from credentials file
import aws_read_and_decrypt_creds 
```

    Successfully defined the following environment variables: BEIWE_SERVER_AWS_ACCESS_KEY_ID, BEIWE_SERVER_AWS_SECRET_ACCESS_KEY, S3_REGION_NAME, S3_BUCKET, STUDY_FOLDER, DECRYPTION_KEY_STRING

```python
# read util functions file
# (assumes we have read credentials file in the line above)
import aws_read_and_decrypt_utils as dcr
```

Define exemplary Beiwe ID value. Note the true value was replaced by "foobar" placeholder for public presentation.

```python
beiwe_id = "foobar0"
```

There are two Beiwe data "groups": chunked and non-chunked. Some Beiwe data streams are stored as chunked data and some other streams as non-chunked data.  

### Browsing chunked data 

To list paths to all data files of type "chunked" for given Beiwe ID, run: 

```python
lpf_out = dcr.list_processed_files(beiwe_id)
lpf_out_l = list(lpf_out)
len(lpf_out_l)
```

    22630

For chunked data, paths for the data files most often have 5 elements and are of the form: `CHUNKED_DATA/<STUDY_NAME>/<beiwe_id>/<Beiwe data stream name>/<file name>`.
    
To learn Beiwe data streams available for this Beiwe ID, we pull 4-th element of the path from each, and get their unique values. 

```python
lpf_datastreams = [k.split("/")[3] for k in lpf_out_l]
list(set(lpf_datastreams))
```

    ['power_state', 'accelerometer', 'magnetometer', 'gps', 'identifiers', 'ios_log', 'proximity', 'gyro', 'reachability', 'survey_timings']

To subset all data files paths to those for accelerometer Beiwe data stream only, run: 

```python
# filter to keep paths with 'accelerometer' phrase only
lpf_out_l_acc = [k for k in lpf_out_l if '/accelerometer/' in k]
len(lpf_out_l_acc)
```

    3865

### Reading in chunked data 

For one file path, to read data from that file from S3, decrypt, and convert into `Pandas` `DataFrame`, run: 

```python
# select one exemplary path of, say, acceleromtry data
fpath = lpf_out_l_acc[123]
# read data from that file from S3, decrypt
download_out = dcr.download(fpath)
# convert bytes stream into Pandas DataFrame
data = pd.read_csv(BytesIO(download_out))
```

The `data` object can be now conveniently used for further processing and analysis, and/or get saved to a local file. 

### Browsing non-chunked data 

To list paths to all data files of type "non-chunked" for given Beiwe ID, run: 

```python
lrf_out = dcr.list_raw_files(beiwe_id)
lrf_out_l = list(lrf_out)
len(lrf_out_l)
```

    19097

For non-chunked data, the data files have paths of different number of path elemements: 

(a) files which path has 2 elements: `<STUDY_NAME>/<file_name>`;
(b) files which path has 3 elements: `<STUDY_NAME>/<beiwe_id>/<file name>`;
(c) files which path has 4 elements: `<STUDY_NAME>/<beiwe_id>/<data stream name>/<file name>`;
(d) files which path has 5 elements: `<STUDY_NAME>/<beiwe_id>/<data stream name>/<data stream subgroup name>/<file name>`.

Typically, 

(a) is the case only for one data file which name is identical with the Beiwe ID; 
(b) is the case only for two data files, so called "identifiers" files;
(d) is the case when there is a data stream subgroup name present, e.g.: separate subdirectories for specific survey instance names for a `surveyTimings` Beiwe data stream. 

To list data streams for files which path has 4 elements, run: 

```python
lrf_datastreams = [k.split("/")[2] for k in lrf_out_l if len(k.split("/")) == 4]
list(set(lrf_datastreams))
```

    ['reachability', 'powerState', 'magnetometer', 'gps', 'proximity', 'gyro', 'accel']

To list data streams for files which path has 5 elements, run: 

```python
lrf_datastreams = [k.split("/")[2]  for k in lrf_out_l if len(k.split("/")) == 5]
list(set(lrf_datastreams))
```

    ['surveyTimings', 'voiceRecording', 'ios', 'surveyAnswers']

To list data streams for files which path has 5 elements, together with the subdirectory name, run: 

```python
lrf_datastreams = [(k.split("/")[2] + "/" + k.split("/")[3])  for k in lrf_out_l if len(k.split("/")) == 5]
list(set(lrf_datastreams))
```

    ['surveyAnswers/bcFG1UiHGEY1KquF7ldgWxm5', 'voiceRecording/1XWTf21WeFP4ZBjzL6t4shGW', 'voiceRecording/6nAmZjcyhgMJ2fbvHlBVHOO6', 'surveyAnswers/QEWH1uOSXXm8wcrJL2qxjf18', 'surveyTimings/f2yTo14DalS0mmSeaTvIOsbM', 'surveyTimings/6nAmZjcyhgMJ2fbvHlBVHOO6', 'surveyAnswers/QjJ2KjHj2hRIfq6KZO8dk07q', 'surveyTimings/rbkORDigMPEKJv0pNyYLQWjr', 'surveyAnswers/xsz8MM4noJMUU79ZPTxU58Xe', 'surveyTimings/QEWH1uOSXXm8wcrJL2qxjf18', 'ios/log', 'surveyTimings/Hv2UrfeStUKOHcx93TCIPj62', 'surveyAnswers/JDi4MtN5tMjzddPm7PHFpEkB', 'surveyTimings/DkTiiOFptQKGOEwOrTDoquuM', 'surveyTimings/j1cdydDGQbiZY5K3fJkkdb2W', 'surveyTimings/eIA3NIw1glKEdQCNQvF1NkFt', 'surveyTimings/QjJ2KjHj2hRIfq6KZO8dk07q', 'surveyTimings/bcFG1UiHGEY1KquF7ldgWxm5', 'surveyAnswers/5o8McpJ5zPd9VeY08giWguE8', 'surveyTimings/FtzsixA8kBHGk6yq3zmbuAoG', 'surveyTimings/1XWTf21WeFP4ZBjzL6t4shGW', 'surveyAnswers/DkTiiOFptQKGOEwOrTDoquuM', 'surveyTimings/ZkRWvnlgQdYJm5VrfamYV3nb', 'surveyAnswers/Q1t0zpGcvjT4Y3XzTOfUZTjV', 'surveyAnswers/ZkRWvnlgQdYJm5VrfamYV3nb', 'surveyTimings/xsz8MM4noJMUU79ZPTxU58Xe', 'voiceRecording/rbkORDigMPEKJv0pNyYLQWjr', 'surveyAnswers/aG3zO2kcaPbaJcUVWfIhR3Lk', 'surveyTimings/aG3zO2kcaPbaJcUVWfIhR3Lk', 'surveyAnswers/f2yTo14DalS0mmSeaTvIOsbM', 'voiceRecording/M8jpf9J0VXvmhVan7pKmfs9f', 'voiceRecording/eIA3NIw1glKEdQCNQvF1NkFt', 'surveyTimings/5o8McpJ5zPd9VeY08giWguE8', 'surveyAnswers/j1cdydDGQbiZY5K3fJkkdb2W', 'surveyTimings/Q1t0zpGcvjT4Y3XzTOfUZTjV', 'surveyTimings/JDi4MtN5tMjzddPm7PHFpEkB', 'surveyAnswers/Hv2UrfeStUKOHcx93TCIPj62']

To subset all data files paths to those for accelerometer Beiwe data stream only, run: 

```python
# filter to keep paths with '/gps/' phrase only
lpf_out_l_gps = [k for k in lpf_out_l if '/gps/' in k]
len(lpf_out_l_gps)
```

    6268

### Reading in non-chunked data 

For one file path, to read data from that file from S3, decrypt, and convert into `Pandas` `DataFrame`, run: 

```python
# select one exemplary path of, say, GPS data
fpath = lpf_out_l_gps[123]
# read data from that file from S3, decrypt
download_out = dcr.download(fpath)
# convert bytes stream into Pandas DataFrame
data = pd.read_csv(BytesIO(download_out))
```

The `data` object can be now conveniently used for further processing and analysis, and/or get saved to a local file. 

### Other notes

#### An exemplary workflow for downloading Beiwe data 

Downloading Beiwe data directly from Amazon S3 into one's machine (local computer or a remote server) could be done with the use of above code. The exemplary workflow could include: (1) iterating over Beiwe IDs, (2) for each Beiwe ID, iterating over all chunked and non-chunked data files, reading in and decrypting them, and saving them into one's machine. 

#### Rare case of a corrupted file 

While iterating over large number of chunked and non-chunked data files, one can encounter a very rare case of a corrupted file. This will likely result in error when executing `data = pd.read_csv(BytesIO(download_out))` part. To account for it, consider wrapping the critical code pieces with `Try Except` clause, for example: 

```python
# (...)
try:
  download_out = dcr.download(fpath)
  data = pd.read_csv(BytesIO(download_out))
except Exception as er:
  print(er)
# (...)
```

## Granting access to Amazon S3 bucket (advanced)

This section contains notes about granting an access to Amazon S3 bucket. This is a system administration content, not needed for a typical target audience of the workflow presented above.  

AWS Identity and Access Management ([IAM](https://aws.amazon.com/iam/)) allows to specify who or what can access services and resources in AWS. 

To grant an AWS user access to the Amazon S3 bucket and objects, in the AWS Online Console go to the IAM service and create a new policy. The example below contains both participant-specific and study-specific example lines (clean up and delete trailing commas as appropriate). Note that AWS has a 6,144 character limit on policy JSON length; if you are doing a Beiwe participant-based restriction you may need to split this up across several policies.

```sh
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": "s3:GetObject",
            "Resource": [
                "arn:aws:s3:::NAME_OF_YOUR_BEIWE_S3_BUCKET",
                "arn:aws:s3:::NAME_OF_YOUR_BEIWE_S3_BUCKET/STUDY_ID*",
                "arn:aws:s3:::NAME_OF_YOUR_BEIWE_S3_BUCKET/CHUNKED_DATA/STUDY_ID*"
            ]
        },
        {
            "Sid": "VisualEditor1",
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": "arn:aws:s3:::NAME_OF_YOUR_BEIWE_S3_BUCKET",
            "Condition": {
                "StringLike": {
                    "s3:prefix": [
                        "STUDY_ID*",
                        "CHUNKED_DATA/STUDY_ID*"
                    ]
                }
            }
        }
    ]
}
```