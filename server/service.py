from flask import Flask, request, jsonify
import numpy as np
from annoy import AnnoyIndex
import os
import json
from query_embeddings import query_embeddings, get_embedding_for_channel, convert_query_to_embedding
from embeddings_generator import process_channel, update_youtube_csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from utils import APIKeyManager, Config
from googleapiclient.discovery import build

config = Config()

youtube_csv = config.get('files', 'youtube_csv')  # Original dataset
youtube_csv_small = config.get('files', 'youtube_csv_small')  # Dataset for new information

embeddings_dir = config.get('folders', 'embeddings_dir')  # Directory where embeddings are stored
index_file = config.get('files', 'index_file')  # File path for the Annoy index
channel_id_map_file = config.get('files', 'channel_id_map')  # File path for the channel ID map


app = Flask(__name__)

# Initialize resources
annoy_index = None
channel_id_map = None
key_manager = APIKeyManager()

# Load the data
df = pd.read_csv(youtube_csv)

# Filter dataframe
embedding_files = set(f.split('.')[0] for f in os.listdir(embeddings_dir) if f.endswith('.json'))
df = df[df['channel_id'].isin(embedding_files)]

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0', 'banner_link', 'avatar', 'total_videos', 'join_date',
                     'mean_views_last_30_videos', 'median_views_last_30_videos', 'std_views_last_30_videos'])

# Create filter options
country_options = [{'label': country, 'value': country} for country in df['country'].dropna().unique()]
subscribers_min = int(df['subscriber_count'].min())
subscribers_max = int(df['subscriber_count'].max())
total_views_min = int(df['total_views'].min())
total_views_max = int(df['total_views'].max())
videos_per_week_min = float(df['videos_per_week'].min())
videos_per_week_max = float(df['videos_per_week'].max())

def apply_filters(df, filters):
    if filters['country']:
        df = df[df['country'].isin(filters['country'])]
    if filters['subscribers']:
        df = df[(df['subscriber_count'] >= filters['subscribers'][0]) & 
               (df['subscriber_count'] <= filters['subscribers'][1])]
    if filters['total_views']:
        df = df[(df['total_views'] >= filters['total_views'][0]) & 
               (df['total_views'] <= filters['total_views'][1])]
    if filters['videos_per_week']:
        df = df[(df['videos_per_week'] >= filters['videos_per_week'][0]) & 
               (df['videos_per_week'] <= filters['videos_per_week'][1])]
    return df

def get_youtube_thumbnail_and_update(channel_id, small_csv=youtube_csv_small):
    def youtube_operation(api_key):
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.channels().list(
            part="snippet,brandingSettings,statistics",
            id=channel_id
        )
        data = request.execute()
        
        if 'items' not in data or not data['items']:
            print(f"No data found for channel ID {channel_id}.")
            return None

        item = data['items'][0]
        snippet = item['snippet']
        medium_thumbnail = snippet.get('thumbnails', {}).get('medium', {}).get('url')

        channel_data = {
            'channel_id': channel_id,
            'channel_link': f"https://www.youtube.com/channel/{channel_id}",
            'channel_name': snippet['title'],
            'subscriber_count': item['statistics'].get('subscriberCount', 'N/A'),
            'banner_link': item.get('brandingSettings', {}).get('image', {}).get('bannerExternalUrl', ''),
            'description': snippet.get('description', ''),
            'keywords': item.get('brandingSettings', {}).get('channel', {}).get('keywords', ''),
            'avatar': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
            'country': snippet.get('country', ''),
            'total_views': item['statistics'].get('viewCount', 'N/A'),
            'total_videos': item['statistics'].get('videoCount', 'N/A'),
            'join_date': snippet['publishedAt'],
            'mean_views_last_30_videos': None,
            'median_views_last_30_videos': None,
            'std_views_last_30_videos': None,
            'videos_per_week': None,
            'last_updated': datetime.now().strftime("%Y-%m-%d")
        }

        update_youtube_csv(channel_data, small_csv)
        return medium_thumbnail

    try:
        return key_manager.try_youtube_operation(youtube_operation)
    except Exception as e:
        print(f"Error retrieving channel data: {e}")
        return None

def load_resources():
    global annoy_index, channel_id_map
    annoy_index = AnnoyIndex(1536, 'angular')
    annoy_index.load(index_file)
    with open(channel_id_map_file, 'r') as f:
        channel_id_map = json.load(f)

def extract_channel_id_from_link(channel_link):
    resp = requests.get(channel_link)
    soup = BeautifulSoup(resp.text, 'html.parser')
    try:
        return soup.select_one('meta[property="og:url"]')['content'].strip('/').split('/')[-1]
    except:
        return None

def generate_channel_embedding(channel_id, output_dir=embeddings_dir):
    file_path = os.path.join(output_dir, f"{channel_id}.json")
    if not os.path.exists(file_path):
        process_channel((channel_id, output_dir))

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    channel_link = data.get('channel_link')
    selected_channels = data.get('selected_channels', [])
    search_query = data.get('search_query')
    filters = data.get('filters', {})
    embeddings_list = []

    if channel_link:
        channel_id = extract_channel_id_from_link(channel_link)
        if channel_id:
            try:
                generate_channel_embedding(channel_id)
                embedding = get_embedding_for_channel(channel_id)
                embeddings_list.append(embedding)
            except Exception as e:
                print(f"Error with channel link: {e}")

    for channel_id in selected_channels:
        try:
            embedding = get_embedding_for_channel(channel_id)
            embeddings_list.append(embedding)
        except Exception as e:
            print(f"Error with selected channel {channel_id}: {e}")

    if search_query:
        try:
            query_embedding = convert_query_to_embedding(search_query)
            if query_embedding is not None:
                embeddings_list.append(query_embedding)
        except Exception as e:
            print(f"Error with search query: {e}")

    if not embeddings_list:
        return jsonify({"error": "No valid input provided."}), 400

    avg_embedding = np.mean(np.array(embeddings_list), axis=0)
    filter_applied = any([filters.get(k) for k in ['country', 'subscribers', 'total_views', 'videos_per_week']])

    if filter_applied:
        filtered_df = apply_filters(df.copy(), filters)
        filtered_channel_ids = filtered_df['channel_id'].tolist()

        if not filtered_channel_ids:
            return jsonify({"error": "No channels match the filters."}), 400

        annoy_index_subset = AnnoyIndex(1536, 'angular')
        subset_channel_id_map = {str(i): channel_id for i, channel_id in enumerate(filtered_channel_ids)}

        for i, channel_id in enumerate(filtered_channel_ids):
            try:
                embedding = get_embedding_for_channel(channel_id)
                annoy_index_subset.add_item(i, embedding)
            except ValueError as e:
                print(f"Skipping channel {channel_id}: {e}")

        annoy_index_subset.build(10)
        results = query_embeddings(avg_embedding, annoy_index_subset, subset_channel_id_map, top_n=100)
    else:
        results = query_embeddings(avg_embedding, annoy_index, channel_id_map, top_n=100)

    response_data = []
    for title, description, channel_id, score in results:
        thumbnail_data = get_youtube_thumbnail_and_update(channel_id)
        response_data.append({
            'title': title,
            'description': description,
            'channel_id': channel_id,
            'score': score,
            'channel_link': f"https://www.youtube.com/channel/{channel_id}",
            'thumbnail': thumbnail_data
        })

    return jsonify(response_data)

@app.route('/filters', methods=['GET'])
def get_filters():
    return jsonify({
        'country_options': country_options,
        'subscribers_min': subscribers_min,
        'subscribers_max': subscribers_max,
        'total_views_min': total_views_min,
        'total_views_max': total_views_max,
        'videos_per_week_min': videos_per_week_min,
        'videos_per_week_max': videos_per_week_max
    })

def extract_channel_id(channel_input):
    for key, value in channel_id_map.items():
        if channel_input == value:
            return value

    matched_row = df[df['channel_name'].str.lower() == channel_input.lower()]
    if not matched_row.empty:
        channel_id = matched_row.iloc[0]['channel_id']
        for key, value in channel_id_map.items():
            if channel_id == value:
                return value
    return None

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    data = request.json
    channel_input = data.get('channel_input')
    prompt = data.get('prompt')

    channel_id = extract_channel_id(channel_input)
    if not channel_id:
        return jsonify({"error": "Invalid channel ID or Name provided"}), 400

    channel_embedding = get_embedding_for_channel(channel_id)
    prompt_embedding = convert_query_to_embedding(prompt, False)
    similarity_score = np.dot(channel_embedding, prompt_embedding)
    channel_details = df[df['channel_id'] == channel_id].iloc[0]

    return jsonify({
        'similarity_score': similarity_score,
        'channel_name': channel_details['channel_name'],
        'channel_id': channel_id
    })

@app.route('/search_channels', methods=['GET'])
def search_channels():
    prefix = request.args.get('prefix', '').lower()
    matching_channels = df[df['channel_name'].str.lower().str.startswith(prefix)]
    channels = [{'label': row['channel_name'], 'value': row['channel_id']} for _, row in matching_channels.iterrows()]
    return jsonify(channels)

if __name__ == '__main__':
    load_resources()
    # Suppress default Flask server messages
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )