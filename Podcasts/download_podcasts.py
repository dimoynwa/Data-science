import requests
import re
from bs4 import BeautifulSoup
import argparse
import os

def print_number_of_results_decorator(func):
    def wrapper_print_number_of_results(*args, **kwargs):
        res = func(*args, **kwargs)
        print('Returned ', len(res), 'number of elements')
        return res
    return wrapper_print_number_of_results

def create_folder(foldername):
    if (not os.path.exists(foldername)) or os.path.isfile(foldername):
        os.mkdir(foldername)
    else:
        print(f'Folder with name {foldername} already exists')

def read_and_parse_html(rss_feed_url):
    # Page content from Website URL
    page = requests.get(rss_feed_url)
    return BeautifulSoup( page.content , 'xml')

@print_number_of_results_decorator
def find_all_items(soup):
    # Get all items
    items = soup.find_all('item')
    return items

def download_and_save(url, filename):
    mp3 = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(mp3.content)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--limit", help="How many podcasts to download")
    parser.add_argument("--regex", help="Regular expression to search for")
    args=parser.parse_args()

    limit = -1
    if args.limit:
        limit = int(args.limit)
    
    regex = '.*'
    if args.regex:
        regex = args.regex

    create_folder('downloads')

    url = 'https://feeds.megaphone.fm/hubermanlab'
    print(f'Start downloading files from {url} with limit {limit} and regex {regex}')
    items = find_all_items(read_and_parse_html(url))
    count = 0
    for item in items:
        title = item.find('title').text
        url = item.find('enclosure')['url']
        description = item.find('description').text
        # We can configure regex pattern for searching in item description
        if re.search(regex, description, re.I):
            print('Title:', title)
            download_and_save(url, 'downloads/'+title+'.mp3')
            print('Downoloaded ', title)
            count += 1
            if count == limit:
                break
