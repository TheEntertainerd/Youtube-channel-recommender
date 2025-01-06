# YouTube Channel Recommender

A recommendation system for discovering YouTube channels. 

<p align="center">
  <img width="465" alt="Screenshot of YouTube Channel Recommender App" src="https://github.com/user-attachments/assets/0481cdfa-79e8-4a62-8d36-2f3803786ed4" />
</p>

<p align="center"><i>Screenshot: Example interface of the Dash webapp</i></p>


## Features

### Embeddings

Embeddings are made for each channel with the openai api, using as informations: name of the channel, description of the channel, previous videos with title, description and a section of the subtitles... These embeddings are then used to perform similarity searches in the Dash web app

### Functionalities of Dash webapp
- Text-based queries (e.g., "cooking channels in Spanish")
- Channel-based similarity (using existing channels from the dataset)
- URL-based search (analyze any YouTube channel from link)
- Hybrid search combining multiple modes (text + channels + URLs). When multiple search criteria are used, the system averages the embeddings to find channels that best match the combined criteria. For example: Searching with "MrBeast" + "channels in Spanish" should find channels that are somewhere in the middle of these constraints
- The system automatically adds new channels to the dataset when searched via URL. While running  this locally is not very useful, it would be a very good feature if it was a website with several users (which is one of my future plan)

### New data Management

-New channels discovered through URL searches are automatically saved in youtube_small.csv
-While this feature primarily demonstrates the system's capability to learn from user interactions, you can merge these new channels with the main dataset using:
```bash
python merge_csv.py
```
- This merging capability would be particularly useful in a multi-user web environment where the system continuously learns from user searches


## Prerequisites

- **Python 3.8+**
- **An OpenAI API Key**
- **YouTube Data API Keys**


## Setup

### 1. API Keys Configuration

Create a `secrets.json` file in the project root:
```json
{
    "openai_api_key": "your-openai-key",
    "youtube_api_keys": [
        "your-youtube-key-1",
        "your-youtube-key-2",
        "your-youtube-key-3"
    ]
}
```

Note: Multiple YouTube API keys are recommended due to the daily quota limits. You can create multiple keys in the same GCP account by creating different projects. Depending on the dataset size, processing might need to be spread across several days.

Links to:
- [OpenAI API Key](https://platform.openai.com/docs/overview)
- [Google cloud console (for Youtube API key)](https://console.cloud.google.com/)

### 2. Build

#### Option A: Build from Scratch

1. Install base requirements:
```bash
pip install -r requirements.txt
```

2. Modify `config.json` if needed
- Adjust video processing settings to choose what information goes into the description of a channel
- Select embedding and completion models

3. Download and process the dataset:
- Get the dataset from [Kaggle](https://www.kaggle.com/datasets/asaniczka/2024-youtube-channels-1-million)
- Place `youtube_channels_1M_clean.csv` in the project root
- Run the following script to filter out channels below a desired number of subscribers:
```bash
python process_kaggle_dataset.py
```

4. Generate embeddings and annoy index for speedup:
```bash
python embeddings_generator.py
python create_annoy_index.py
```

#### Option B: Use existing data (200k+ Subscribers)

1. Download the pre-processed data package from [Google Drive Link](https://drive.google.com/file/d/1fvpNyhu2ta98_UlM5yw-Z6g27NawGySj/view?usp=drive_link)
2. Extract the contents directly to the project root
3. Verify the following files are present:
   - `embeddings/` folder with around 40000 embeddings
   - `index.ann`
   - `channel_id_map.json`
   - `youtube.csv`

### 3. Running the Application

#### Option A: Using Docker Compose (Recommended)

```bash
docker-compose up --build
```
Access at http://localhost:8050

#### Option B: Using Virtual Environment

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Start the services (in separate terminals):
```bash
# Terminal 1
python server/service.py

# Terminal 2
python dash/app.py
```

Access at http://localhost:8050

### 4. Using the Application

- The application provides a single interface combining all search modalities
- With the default dataset (~40,000 channels), queries typically take about 4 seconds on my mid-range laptop but it may increase with larger datasets

## Project Structure

```
├── dash/
│   └── app.py              # Frontend Dash application
├── server/
│   ├── service.py          # Backend Flask 
│   └── query_embeddings.py
├── embeddings/             # Channel embeddings
├── config.json             # Configuration 
├── create_annoy_index.py   # Index creation 
├── embeddings_generator.py # Embedding Creation 
├── process_kaggle_dataset.py # Dataset reduction 
├── index.ann              # Annoy index 
├── channel_id_map.json    # Mapping file
├── youtube.csv            # Processed data 
├── requirements.txt       # Project dependencies
├── utils.py              # Utility functions
├── merge_csv.py           # Merge new data into csv
└── secrets.json          # API keys configuration
```
