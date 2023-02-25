import os
import json
import boto3
import time
import pandas as pd
from botocore.exceptions import ClientError

from utils import create_bucket, mp3_files, modify_file_name, save_to_pickle
from download_podcasts import create_folder

class AwsTranscriber:
    def __init__(self, credentials_file) -> None:
        filename = credentials_file or "aws_credentials.json"
        self._read_credentials(filename)

    def upload_files(self):
        s3_client = boto3.client('s3',
            aws_access_key_id = self._access_key,
            aws_secret_access_key = self._secret_access_key,
            region_name = self._region)
        create_bucket(s3_client, self._bucket_name, self._region)
        for file_name in mp3_files('downloads'):
            # Upload the file
            object_name = modify_file_name(os.path.basename(file_name))
            try:
                response = s3_client.upload_file(file_name, self._bucket_name, object_name)
                print(f'Response from uploading {object_name}: {response}')
                yield object_name
            except ClientError as e:
                print('ERROR uploading file', file_name, 'to bucket', self._bucket_name, e)

    def start_transcribe(self, name_generator):
        self._transcribe = boto3.client('transcribe',
            aws_access_key_id = self._access_key,
            aws_secret_access_key = self._secret_access_key, 
            region_name = self._region)
        for filename in name_generator:
            cleaned_f = modify_file_name(filename)
            file_uri = 's3://' + self._bucket_name + '/' + cleaned_f
            
            job_name = ''.join(filter(str.isalpha, cleaned_f)) + str(round(time.time()*1000))
            print('Start transcribe job for ', file_uri, 'job name:', job_name)
            file_format = filename.split('.')[-1]
            self._transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': file_uri},
                MediaFormat = file_format,
                LanguageCode='en-US')
            yield job_name

    def wait_results(self, jobs_generator):
        jobs = {job for job in jobs_generator}
        completed_jobs = set()
        while len(jobs) > len(completed_jobs):
            for job in {j for j in jobs if not j in completed_jobs }:
                result = self._transcribe.get_transcription_job(TranscriptionJobName=job)
                status = result['TranscriptionJob']['TranscriptionJobStatus']
                print(f'Status: {status} for job {job}')
                if status == 'FAILED':
                    print(f'Job with name {job} failed')
                    completed_jobs.add(job)
                if status == 'COMPLETED':
                    print(f'Job with name {job} completed')
                    completed_jobs.add(job)
                    res_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    yield (job, pd.read_json(res_uri))
            time.sleep(10)
         
    def _read_credentials(self, file):
        with open(file, 'r') as f:
            credentials = json.load(f)

            self._access_key = credentials['aws_key']
            self._secret_access_key = credentials['aws_secret_key']
            self._region = credentials['aws_region']
            self._bucket_name = credentials['aws_s3_bucket_name']

if __name__ == '__main__':
    aws_transcriber = AwsTranscriber('aws_credentials.json')
    upload = aws_transcriber.upload_files()
    transcriptions = aws_transcriber.start_transcribe(upload)
    
    transcriptions_folder = 'aws_transcriptions'
    create_folder(transcriptions_folder)

    count = 0
    for transcription in aws_transcriber.wait_results(transcriptions):
        print(f'Transcription: {transcription}')
        count += 1
        save_to_pickle(transcriptions_folder + '/' + transcription[0], transcription[1])
    print(f'Saved {count} transcrptions')