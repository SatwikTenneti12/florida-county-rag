"""
LLM client for RAG generation.

Provides chat completion capabilities for generating answers from retrieved context.
"""

import time
import requests
from typing import List, Dict, Optional
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class LLMClient:
    """Client for LLM chat completions."""
    
    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        timeout: int = 180
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        
        if not self.api_key or not self.base_url:
            raise ValueError("Missing API key or base URL for LLM")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        max_retries: int = 6,
        backoff_seconds: float = 3.0,
    ) -> str:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt to prepend
            
        Returns:
            The assistant's response text
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                # Rate limit / overload: retry with backoff
                if resp.status_code == 429:
                    if attempt >= max_retries:
                        raise RuntimeError(f"LLM API error 429: {resp.text}")
                    wait = backoff_seconds * attempt
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text}")

                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                # Transient network/connection issues: retry
                if attempt >= max_retries:
                    raise RuntimeError(f"LLM request failed after retries: {e}")
                wait = backoff_seconds * attempt
                time.sleep(wait)

        raise RuntimeError("LLM request failed unexpectedly after retries")
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Simple single-turn completion."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature, max_tokens, system_prompt)


# Singleton instance
_client = None

def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
