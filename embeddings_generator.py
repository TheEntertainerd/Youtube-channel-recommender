import os
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI
import openai
from youtube_transcript_api import YouTubeTranscriptApi
import random
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from datetime import datetime
import country_converter as coco
from utils import APIKeyManager, Config


config = Config()

youtube_csv = config.get('files', 'youtube_csv')  # Original dataset
youtube_csv_small = config.get('files', 'youtube_csv_small')  # Dataset for new information

LIMIT_WORDS_VIDEO_SUBTITLES = config.get('video_processing', 'limit_words_video_subtitles')  # Word limit for video subtitles
LIMIT_CHAR_VIDEO_DESCRIPTION = config.get('video_processing', 'limit_char_video_description')  # Character limit for video descriptions
NB_VIDEOS_TO_USE = config.get('video_processing', 'nb_videos_to_use')  # Number of videos to use

model_completion = config.get('models', 'completion')  # Model for description generation
model_embedding = config.get('models', 'embedding')  # Model for computing embeddings

# Load the dataset
df = pd.read_csv(youtube_csv)

def init_worker():
    global key_manager
    key_manager = APIKeyManager()

def parse_channel_info(response_item):
    cc = coco.CountryConverter()

    channel_id = response_item['id']
    channel_link = f"https://www.youtube.com/channel/{channel_id}"
    channel_name = response_item['snippet']['title']
    subscriber_count = response_item['statistics'].get('subscriberCount', 'N/A')
    banner_link = response_item.get('brandingSettings', {}).get('image', {}).get('bannerExternalUrl', '')
    description = response_item['snippet']['description']
    keywords = response_item.get('brandingSettings', {}).get('channel', {}).get('keywords', '')
    avatar = response_item['snippet']['thumbnails']['high']['url']
    country_code = response_item['snippet'].get('country', '')
    country_name = cc.convert(names=country_code, to='name_short')

    total_views = response_item['statistics'].get('viewCount', 'N/A')
    total_videos = response_item['statistics'].get('videoCount', 'N/A')
    join_date = response_item['snippet']['publishedAt']
    last_updated = datetime.now().strftime("%Y-%m-%d")

    return {
        'channel_id': channel_id,
        'channel_link': channel_link,
        'channel_name': channel_name,
        'subscriber_count': subscriber_count,
        'banner_link': banner_link,
        'description': description,
        'keywords': keywords,
        'avatar': avatar,
        'country': country_name,
        'total_views': total_views,
        'total_videos': total_videos,
        'join_date': join_date,
        'mean_views_last_30_videos': None,
        'median_views_last_30_videos': None,
        'std_views_last_30_videos': None,
        'videos_per_week': None,
        'last_updated': last_updated
    }

def fetch_channel_info(key_manager, channel_id):
    def operation(api_key):
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.channels().list(
            part="snippet,brandingSettings,status,contentOwnerDetails,statistics,topicDetails,localizations,contentDetails,id",
            id=channel_id
        )
        return request.execute()
    
    return key_manager.try_youtube_operation(operation)

def fetch_video_ids(key_manager, channel_id):
    def operation(api_key):
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.channels().list(part="contentDetails", id=channel_id)
        response = request.execute()
        uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=10
        )
        response = request.execute()
        return [item['contentDetails']['videoId'] for item in response['items']]
        
    return key_manager.try_youtube_operation(operation)

def fetch_video_details(key_manager, video_ids):
    def get_video_info(api_key):
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.videos().list(
            part="snippet",
            id=",".join(video_ids)
        )
        return request.execute()

    if not video_ids:
        return []

    response = key_manager.try_youtube_operation(get_video_info)
    
    video_titles = []
    video_descriptions = []
    for item in response['items']:
        video_titles.append(item['snippet']['title'])
        video_descriptions.append(item['snippet']['description'][:LIMIT_CHAR_VIDEO_DESCRIPTION]) # Length limit of description

    subtitle_lists = []
    videos_with_subtitles = 1
    video_details = []

    for i, video_id in enumerate(video_ids):
        if videos_with_subtitles >= NB_VIDEOS_TO_USE:
            break
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
            subtitle_text = " ".join([item['text'] for item in transcript.fetch()])

            words = subtitle_text.split()
            if len(words) > LIMIT_WORDS_VIDEO_SUBTITLES:
                start_index = random.randint(0, len(words) - LIMIT_WORDS_VIDEO_SUBTITLES)
                subtitle_text = " ".join(words[start_index:start_index + LIMIT_WORDS_VIDEO_SUBTITLES]) # Extract a random part of the subtitles

            subtitle_lists.append(subtitle_text)
            video_details.append({
                "title": video_titles[i],
                "description": video_descriptions[i],
                "subtitles": subtitle_text
            })
            videos_with_subtitles += 1
        except Exception:
            pass

    return video_details

def update_youtube_csv(channel_data, csv_file='youtube_small.csv'):
    """
    Update the YouTube CSV file with new channel data.
    
    Args:
        channel_data (dict): Dictionary containing channel information
        csv_file (str): Path to the CSV file to update
    """
    expected_columns = [
        'channel_id', 'channel_link', 'channel_name',
        'subscriber_count', 'banner_link', 'description', 'keywords', 
        'avatar', 'country', 'total_views', 'total_videos',
        'join_date', 'mean_views_last_30_videos',
        'median_views_last_30_videos', 'std_views_last_30_videos',
        'videos_per_week', 'last_updated'
    ]

    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        df_existing = pd.read_csv(csv_file)
    else:
        df_existing = pd.DataFrame(columns=expected_columns)

    df_new = pd.DataFrame([channel_data])

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    df_combined.drop_duplicates(subset='channel_id', keep='last', inplace=True)

    df_combined.to_csv(csv_file, index=False)

    
def generate_description(key_manager, channel_data, video_details):
    video_details_text = "\n\n".join(
        [f"--- Video {i+1} ---\nTitle: {detail['title']}\nDescription: {detail['description']}\nSubtitles:\n{detail['subtitles']}"
        for i, detail in enumerate(video_details[:NB_VIDEOS_TO_USE])]
    )

    prompt = (
        f"Using the information provided, generate a factual and concise description for a YouTube channel. "
        f"The description should emphasize the core content and topics covered by the channel, highlight the type of audience the content is best suited for, "
        f"and avoid promotional language. Make sure the description is informative and to the point, providing potential viewers with a clear understanding of what they can expect from the channel.\n\n"
        f"**Channel Title**: {channel_data['channel_name']}\n"
        f"**Channel Description**: {channel_data['description']}\n"
        f"**Subscriber Count**: {channel_data['subscriber_count']}\n"
        f"**View Count**: {channel_data['total_views']}\n"
        f"**Video Count**: {channel_data['total_videos']}\n"
        f"**Country**: {channel_data['country']}\n"
        f"**Channel Creation Date**: {channel_data['join_date']}\n"
        f"**Video Details**:\n{video_details_text}\n\n"
        f"Focus on these aspects:\n"
        f"1. **Core Content**: Provide a clear outline of the specific topics and themes covered by the channel. "
        f"Mention any key series or regular content segments.\n"
        f"2. **Audience**: Specify the type of viewer who would most benefit from or enjoy the channel.\n"
        f"3. **Style and Tone**: Describe the tone of the content and the approach taken in presenting the content. "
        f"Mention any specific formats.\n"
        f"4. **Language**: Clearly indicate the primary language of the content."
    )

    try:
        client = OpenAI(api_key=key_manager.get_openai_key())
        response = client.chat.completions.create(
            model=model_completion,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred while generating the description: {e}")
        return None

def generate_embeddings(key_manager, description):
    try:
        client = OpenAI(api_key=key_manager.get_openai_key())
        embedding_response = client.embeddings.create(
            input=[description],
            model=model_embedding
        )
        return embedding_response.data[0].embedding
    except Exception as e:
        print(f"An error occurred while computing embeddings: {e}")
        return None

def save_embedding(channel_id, channel_name, description, raw_description, embedding, output_dir="embeddings"):
    embedding_data = {
        "channel_id": channel_id,
        "channel_title": channel_name,
        "generated_description": description,
        "raw_description": raw_description,
        "embeddings": embedding
    }
    file_path = os.path.join(output_dir, f"{channel_id}.json")
    with open(file_path, "w") as f:
        json.dump(embedding_data, f)

def process_channel(args):
    if 'key_manager' not in globals():
        init_worker()

    channel_id, output_dir = args
    file_path = os.path.join(output_dir, f"{channel_id}.json")
    if os.path.exists(file_path):
        print(f"Embedding already exists for channel {channel_id}, skipping...")
        return None

    try:
        response = fetch_channel_info(key_manager, channel_id)
        if 'items' in response and response['items']:
            channel_data = parse_channel_info(response['items'][0])
        else:
            print(f"No data found for channel ID {channel_id}.")
            return None
    except Exception as e:
        print(f"An error occurred while processing channel {channel_id}: {e}")
        return None

    try:
        video_ids = fetch_video_ids(key_manager, channel_id)
        video_details = fetch_video_details(key_manager, video_ids)

        description = generate_description(key_manager, channel_data, video_details)
        if not description:
            return None

        embedding = generate_embeddings(key_manager, description)
        if not embedding:
            return None

        save_embedding(channel_id, channel_data['channel_name'], description, 
                      channel_data['description'], embedding, output_dir)
        
        return channel_data
    except Exception as e:
        print(f"Error processing channel {channel_id}: {e}")
        return None


def main_process(output_dir="embeddings", num_processes=None, csv_file=youtube_csv_small):
    if num_processes is None:
        num_processes = mp.cpu_count()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = [df.iloc[idx] for idx in range(0,1325)]
    tasks = [(row['channel_id'], output_dir) for row in rows]

    with Pool(processes=num_processes, initializer=init_worker) as pool:
        results = []
        for channel_data in tqdm(pool.imap_unordered(process_channel, tasks), total=len(tasks)):
            if channel_data:
                results.append(channel_data)

    for channel_data in results:
        update_youtube_csv(channel_data, csv_file)

if __name__ == "__main__":
    main_process()