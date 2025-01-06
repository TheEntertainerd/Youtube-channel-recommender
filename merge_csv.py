import pandas as pd
import os
from utils import Config

config = Config()

youtube_csv = config.get('files', 'youtube_csv')  # Original dataset
youtube_csv_small = config.get('files', 'youtube_csv_small')  # Dataset for new information

def merge_youtube_data(main_csv=youtube_csv, small_csv=youtube_csv_small):
    if not os.path.exists(main_csv):
        print(f"{main_csv} does not exist.")
        return

    if not os.path.exists(small_csv):
        print(f"{small_csv} does not exist.")
        return

    # Load both datasets
    main_df = pd.read_csv(main_csv)
    small_df = pd.read_csv(small_csv)

    # Merge datasets, giving preference to the updates in small_csv
    merged_df = pd.concat([main_df, small_df]).drop_duplicates(subset='channel_id', keep='last')

    # Save the merged data back to youtube.csv
    merged_df.to_csv(main_csv, index=False)
    print(f"{main_csv} has been updated with new data from {small_csv}.")

    # Delete the small_csv file after merging
    os.remove(small_csv)
    print(f"{small_csv} has been deleted.")

if __name__ == "__main__":
    merge_youtube_data()