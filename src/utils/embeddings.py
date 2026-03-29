"""
Shared embedding utilities for the RAG system.

Provides a consistent interface for generating embeddings via the API.
"""

import requests
from typing import List
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config import EMBEDDINGS_API_KEY, EMBEDDINGS_BASE_URL, EMBEDDINGS_MODEL


class EmbeddingClient:
    """Client for generating text embeddings via API."""
    
    def __init__(
        self,
        api_key: str = EMBEDDINGS_API_KEY,
        base_url: str = EMBEDDINGS_BASE_URL,
        model: str = EMBEDDINGS_MODEL,
        timeout: int = 120
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        
        if not self.api_key or not self.base_url:
            raise ValueError("Missing API key or base URL for embeddings")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text}")
        
        data = resp.json()
        embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed([text])[0]


# Singleton instance for convenience
_client = None

def get_embedding_client() -> EmbeddingClient:
    """Get or create the singleton embedding client."""
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_client().embed(texts)


def embed_query(text: str) -> List[float]:
    """Convenience function to embed a single query."""
    return get_embedding_client().embed_query(text)
