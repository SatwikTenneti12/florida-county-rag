import os
from dotenv import load_dotenv
load_dotenv()

import requests

base = os.environ["EMBEDDINGS_BASE_URL"].rstrip("/")
key = os.environ["EMBEDDINGS_API_KEY"]

url = base + "/models"   # many gateways support /models
r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=60)
print("Status:", r.status_code)
print(r.text[:500])