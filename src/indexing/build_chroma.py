"""
Build a local Chroma vector index from chunks.jsonl using an embedding API.

Inputs:
- data/processed/chunks.jsonl

Outputs:
- chroma_db/  (persistent Chroma store)

ENV required (via .env):
- EMBEDDINGS_API_KEY
- EMBEDDINGS_BASE_URL
- EMBEDDINGS_MODEL (default: nomic-embed-text-v1.5)

Usage:
python src/indexing/build_chroma.py --limit 200
python src/indexing/build_chroma.py
"""

import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout
from tqdm import tqdm
import chromadb
from chromadb.config import Settings


ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "county_chunks"

DEFAULT_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text-v1.5")


# ✅ FIXED: Now includes text hash to prevent duplicate IDs
def stable_id(county: str, pdf_file: str, page_start: int, page_end: int, text: str) -> str:
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    raw = f"{county}||{pdf_file}||{page_start}||{page_end}||{text_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_chunks(limit: int | None = None) -> List[Dict[str, Any]]:
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chunks.append(r)
            if limit and len(chunks) >= limit:
                break
    return chunks


def embed_texts(
    texts: List[str],
    base_url: str,
    api_key: str,
    model: str,
    max_retries: int = 5,
    backoff_seconds: float = 5.0,
) -> List[List[float]]:
    """
    Generate embeddings with simple retry logic for rate limits / transient errors.
    """
    url = base_url.rstrip("/") + "/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": texts
    }

    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)

            # Handle rate limits / overload gracefully
            if resp.status_code == 429:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"Embeddings API error 429 after retries: {resp.text}")
                wait = backoff_seconds * attempt
                print(f"[embed_texts] 429 rate limit/overloaded. Sleeping {wait:.1f}s and retrying...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise RuntimeError(f"Embeddings API error {resp.status_code}: {resp.text}")

            data = resp.json()
            embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            return embeddings

        except (RequestsConnectionError, RequestsTimeout) as e:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Embeddings request failed after retries: {e}")
            wait = backoff_seconds * attempt
            print(f"[embed_texts] Connection error '{e}'. Sleeping {wait:.1f}s and retrying...")
            time.sleep(wait)


def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Embed only first N chunks (0 = all)")
    parser.add_argument("--batch", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between batches")
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")

    base_url = os.getenv("EMBEDDINGS_BASE_URL", "").strip()
    api_key = os.getenv("EMBEDDINGS_API_KEY", "").strip()
    model = os.getenv("EMBEDDINGS_MODEL", DEFAULT_MODEL).strip()

    if not base_url or not api_key:
        raise RuntimeError(
            "Missing env vars:\n"
            "  EMBEDDINGS_BASE_URL\n"
            "  EMBEDDINGS_API_KEY\n"
        )

    limit = args.limit if args.limit and args.limit > 0 else None
    chunks = load_chunks(limit=limit)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Embeddings model: {model}")
    print(f"Chroma dir: {CHROMA_DIR}")

    col = get_chroma_collection()

    # Generate stable IDs (duplicate-safe)
    id_chunk_pairs: List[Tuple[str, Dict[str, Any]]] = []

    for r in chunks:
        cid = stable_id(
            r["county"],
            r["pdf_file"],
            int(r["page_start"]),
            int(r["page_end"]),
            r["text"]
        )
        id_chunk_pairs.append((cid, r))

    # Remove duplicate IDs within current batch
    unique_pairs = {}
    for cid, r in id_chunk_pairs:
        unique_pairs[cid] = r

    id_chunk_pairs = list(unique_pairs.items())

    print(f"Unique chunks to process: {len(id_chunk_pairs)}")

    batch_size = max(1, args.batch)

    for i in tqdm(range(0, len(id_chunk_pairs), batch_size), desc="Embedding+upserting"):
        batch = id_chunk_pairs[i:i + batch_size]

        ids = [cid for cid, _ in batch]
        docs = [r["text"] for _, r in batch]
        metas = [
            {
                "county": r["county"],
                "pdf_file": r["pdf_file"],
                "page_start": int(r["page_start"]),
                "page_end": int(r["page_end"]),
            }
            for _, r in batch
        ]

        embeddings = embed_texts(
            docs,
            base_url=base_url,
            api_key=api_key,
            model=model
        )

        col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\nDONE ✅")
    print("Collection:", COLLECTION_NAME)
    print("Total vectors now:", col.count())


if __name__ == "__main__":
    main()