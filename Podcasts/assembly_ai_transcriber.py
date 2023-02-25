import json
import requests
import time
import argparse
from utils import mp3_files, save_to_pickle
from download_podcasts import create_folder

upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
polling_endpoint = "https://api.assemblyai.com/v2/transcript"

class AssemblyAiTranscriber():
    def __init__(self, credentials_file, time_limit=None) -> None:
        filename = credentials_file or "assembly_ai_credentials.json"
        self._read_credentials(filename)
        self.time_limit = time_limit

    def upload_files(self):
        header = {
	        'authorization': self._api_key,
	        'content-type': 'application/json'
        }
        for file_name in mp3_files('downloads'):
            print(f'Uploading {file_name}...')
            # Upload the file 
            upload_response = requests.post(
                upload_endpoint,
                headers=header, data=self._read_file(file_name)
            )
            yield upload_response.json()['upload_url']

    def start_transcribe(self, name_generator):
        header = {
	        'authorization': self._api_key,
	        'content-type': 'application/json'
        }
        for upload_url in name_generator: 
            transcript_request = {
                'audio_url': upload_url
            }
            if self.time_limit:
                transcript_request['audio_start_from'] = 0
                transcript_request['audio_end_at'] = self.time_limit * 1000
            transcript_response = requests.post(
                transcript_endpoint,
                json=transcript_request,
                headers=header
            )
            yield transcript_response.json()['id']

    def wait_results(self, jobs_generator):
        header = {
	        'authorization': self._api_key,
	        'content-type': 'application/json'
        }

        jobs = {job for job in jobs_generator}
        completed_jobs = set()
        while len(jobs) > len(completed_jobs):
            for job in {j for j in jobs if not j in completed_jobs }:
                endpoint = polling_endpoint + f'/{job}'
                polling_response = requests.get(endpoint, headers=header)
                polling_response = polling_response.json()

                print('Polling status:' + polling_response['status'])
                if polling_response['status'] == 'completed':
                    completed_jobs.add(job)
                    yield (job, self._get_paragraphs(endpoint, header))
            time.sleep(5)

    def _get_paragraphs(self, polling_endpoint, header):
        paragraphs_response = requests.get(polling_endpoint + "/paragraphs", headers=header)
        paragraphs_response = paragraphs_response.json()

        paragraphs = [para['text'] for para in paragraphs_response['paragraphs']]

        return ''.join(paragraphs)

    def _read_credentials(self, filename):
        with open(filename, 'r') as f:
            credentials = json.load(f)    
        self._api_key = credentials['assemly_ai_api_key']

    def _read_file(self, filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--limit", help="How many seconds of podcast to transcribe")
    args=parser.parse_args()

    limit = None
    if args.limit:
        limit = int(args.limit)
    
    assemblyAiTranscriber = AssemblyAiTranscriber('assembly_ai_credentials.json', time_limit=limit)
    
    transcriptions_folder = 'assembly_ai_transcriptions'
    create_folder(transcriptions_folder)
    
    upload = assemblyAiTranscriber.upload_files()
    transcriptions = assemblyAiTranscriber.start_transcribe(upload)
    count = 0
    for transcription in assemblyAiTranscriber.wait_results(transcriptions):
        count += 1
        print(f'Transcription: {transcription}')
        save_to_pickle(transcriptions_folder + '/' + transcription[0], transcription[1])
    print(f'Saved {count} transcrptions')