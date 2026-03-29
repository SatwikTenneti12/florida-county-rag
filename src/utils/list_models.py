"""
List available models from the API endpoint.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("EMBEDDINGS_API_KEY", "").strip()
BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "").strip()


def list_models():
    """Query the API for available models."""
    url = f"{BASE_URL.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            print("Available models:\n")
            for model in data.get("data", []):
                model_id = model.get("id", "unknown")
                owned_by = model.get("owned_by", "")
                print(f"  - {model_id}")
                if owned_by:
                    print(f"    (owned by: {owned_by})")
            return data
        else:
            print(f"Error {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        print(f"Failed to list models: {e}")
        return None


if __name__ == "__main__":
    list_models()
