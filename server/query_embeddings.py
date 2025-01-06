import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import time
from annoy import AnnoyIndex
import sys
# Set project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from utils import APIKeyManager, Config

config = Config()

youtube_csv = config.get('files', 'youtube_csv')  # Original dataset
youtube_csv_small = config.get('files', 'youtube_csv_small')  # Dataset for new information

embeddings_dir = config.get('folders', 'embeddings_dir')  # Directory where embeddings are stored
index_file = config.get('files', 'index_file')  # File path for the Annoy index
channel_id_map = config.get('files', 'channel_id_map')  # File path for the channel ID map

model_embedding = config.get('models', 'embedding')  # Model for computing embeddings

# Set up OpenAI API client
key_manager = APIKeyManager()
OPENAI_API_KEY = key_manager.get_openai_key()
client = OpenAI(api_key=OPENAI_API_KEY)

threshold = 0.55


def rephrase_user_query(query):
    """Rephrase the user query to match the style of the generated descriptions."""
    try:
        prompt = (
            f"Rephrase the following user query to match the style of a structured, factual, and concise YouTube channel description. "
            f"The goal is to align the query with the style of an informative overview.\n\n"
            f"Translate in English if necessary.\n\n"
            f"Original Query: '{query}'\n\n"
            f"Rephrased Query:"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        rephrased_query = response.choices[0].message.content.strip()
        return rephrased_query
    
    except Exception as e:
        print(f"An error occurred while rephrasing the query: {e}")
        return query  # Fall back to the original query if something goes wrong
    

def get_embedding_for_channel(channel_id, embedding_dir=embeddings_dir):
    embedding_file = os.path.join(embedding_dir, f"{channel_id}.json")
    if os.path.exists(embedding_file):
        with open(embedding_file, "r") as f:
            data = json.load(f)
            return np.array(data['embeddings'])
    else:
        raise ValueError(f"Embedding for channel_id {channel_id} not found.")


def load_annoy_index(index_file=index_file, channel_id_map_json=channel_id_map, embedding_dir=embeddings_dir):
    with open(channel_id_map, "r") as f:
        channel_id_map_json = json.load(f)
    
    example_file = next(f for f in os.listdir(embedding_dir) if f.endswith(".json"))
    with open(os.path.join(embedding_dir, example_file), "r") as f:
        example_data = json.load(f)
        embedding_dim = len(example_data['embeddings'])

    annoy_index = AnnoyIndex(embedding_dim, 'angular')
    annoy_index.load(index_file)
    print(embedding_dim,annoy_index,example_data,embedding_dim)
    
    return annoy_index, channel_id_map_json



def convert_query_to_embedding(query,rephrase_query=True):
    rephrased_query = rephrase_user_query(query)
    print(f"Rephrased Query: {rephrased_query}")
    
    try:
        query_embedding_response = client.embeddings.create(
            input=[rephrased_query],
            model=model_embedding
        )
        query_embeddings = query_embedding_response.data[0].embedding
        return query_embeddings
    except Exception as e:
        print(f"An error occurred while computing embeddings for the query: {e}")
        return None


def query_embeddings(embedding, annoy_index, channel_id_map, top_n=10):
    
    query_embeddings_np = embedding

    # Perform nearest neighbor search using Annoy
    indices, distances = annoy_index.get_nns_by_vector(query_embeddings_np, top_n, include_distances=True)
    
    results = []
    for idx, distance in zip(indices, distances):
        similarity_score = 1 - distance / 2  # Convert distance to similarity
        if similarity_score >= threshold:
            channel_id = channel_id_map[str(idx)]
            with open(os.path.join(embeddings_dir, f"{channel_id}.json"), "r") as f:
                data = json.load(f)
                results.append((data['channel_title'], data['raw_description'], channel_id, similarity_score))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:13] # Return top 13 results

if __name__ == "__main__":
    annoy_index, channel_id_map = load_annoy_index()
    
    # Example usage with a query string
    query = "A channel that teaches about insects"
    results = query_embeddings(query, annoy_index, channel_id_map)

    # Example usage with a direct embedding
    channel_id = "some_channel_id"
    embedding = get_embedding_for_channel(channel_id)
    results = query_embeddings(embedding, annoy_index, channel_id_map)

    for title, description, channel_id, score in results:
        print(f"Channel: {title}\nDescription: {description}\nChannel ID: {channel_id}\nSimilarity Score: {score}\n")
