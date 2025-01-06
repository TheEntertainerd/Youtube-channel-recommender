import os
import json
import numpy as np
from tqdm import tqdm
from annoy import AnnoyIndex
from utils import Config


config = Config()


embeddings_dir = config.get('folders', 'embeddings_dir')  # Directory where are stored embeddings
index_file = config.get('files', 'index_file')  # File path for the Annoy index
channel_id_map = config.get('files', 'channel_id_map')  # File path for the channel ID map

def build_annoy_index(embedding_dir=embeddings_dir, index_file=index_file, channel_id_map_file=channel_id_map, trees=10):
    """
    Build an Annoy index from embedding files stored in the specified directory.

    Args:
        embedding_dir (str): Directory containing embedding JSON files.
        index_file (str): Output file path for the Annoy index.
        trees (int): Number of trees to build the Annoy index with.
    """
    # Check if the embedding directory exists
    if not os.path.exists(embedding_dir):
        print(f"Embedding directory '{embedding_dir}' not found. Exiting.")
        return

    # Determine the embedding dimension from the first file
    try:
        example_file = next(f for f in os.listdir(embedding_dir) if f.endswith(".json"))
        with open(os.path.join(embedding_dir, example_file), "r") as f:
            example_data = json.load(f)
            embedding_dim = len(example_data['embeddings'])
    except StopIteration:
        print(f"No JSON files found in the embedding directory '{embedding_dir}'. Exiting.")
        return
    except KeyError:
        print("Embeddings not found in the example file. Ensure your JSON files contain 'embeddings' key.")
        return

    # Initialize Annoy index
    annoy_index = AnnoyIndex(embedding_dim, 'angular')

    # Map to store channel_id for each index in Annoy
    channel_id_map = {}

    # Add items to the Annoy index
    i = 0
    for filename in tqdm(os.listdir(embedding_dir), desc="Processing embeddings"):
        if filename.endswith(".json"):
            channel_id = filename.split('.')[0]
            file_path = os.path.join(embedding_dir, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    embedding = np.array(data['embeddings'])
                    annoy_index.add_item(i, embedding)
                    channel_id_map[i] = channel_id
                    i += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing file {filename}: {e}")

    # Build the Annoy index
    print("Building Annoy index...")
    annoy_index.build(trees)
    annoy_index.save(index_file)

    # Save the channel ID map
    with open(channel_id_map_file, "w") as f:
        json.dump(channel_id_map, f)

if __name__ == "__main__":
    build_annoy_index()
