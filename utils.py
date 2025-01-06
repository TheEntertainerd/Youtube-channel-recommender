import json
from googleapiclient.errors import HttpError
import os
import pandas as pd


class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from config.json file."""
        try:
            with open('config.json', 'r') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("config.json not found. Please ensure it exists in the root directory.")
        except json.JSONDecodeError:
            raise ValueError("config.json is not a valid JSON file.")

    def get(self, *keys: str, default= None):

        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


    

class APIKeyManager:
    def __init__(self, secrets_file: str = "secrets.json"):
        with open(secrets_file) as f:
            self.secrets = json.load(f)
            
        self.youtube_keys = self.secrets["youtube_api_keys"]
        if not self.youtube_keys:
            raise ValueError("No YouTube API keys found in secrets.")
            
        self.openai_key = self.secrets.get("openai_api_key")

    def try_youtube_operation(self, operation):

        last_error = None
        for key in self.youtube_keys:
            try:
                return operation(key)
            except HttpError as e:
                if e.resp.status not in [400, 403, 429]:  # If not a quota/auth error, raise immediately
                    raise
                last_error = e
                continue
                
        raise Exception(f"All YouTube API keys exhausted. Last error: {last_error}")

    def get_openai_key(self) -> str:
        return self.openai_key
    