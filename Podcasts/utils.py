import os
import pathlib
import pickle
from botocore.exceptions import ClientError

def create_bucket(s3_client, bucket_name, region=None):
    buckets_response = s3_client.list_buckets()
    for bucket in buckets_response['Buckets']:
        if bucket['Name'] == bucket_name:
            print('Bucket with name', bucket_name, 'already exists. Skip creating')
            return True
    try:
        if region is None:
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        print('Error creating bucket', e)
        return False
    return True

def mp3_files(dir_name):
    for filename in os.listdir(dir_name):
        file_extension = pathlib.Path(filename).suffix
        if file_extension == '.mp3':
            yield os.path.join(dir_name, filename)

def modify_file_name(filename):
    modified_file_name = filename.replace(' ', '-').replace(':', '')
    modified_file_name = modified_file_name.lower()
    return modified_file_name

def save_to_pickle(filename, data):
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)