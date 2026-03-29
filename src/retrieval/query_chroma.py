"""
Query the local Chroma DB with an embedding API and print citation-ready results.

ENV (from .env):
- EMBEDDINGS_API_KEY
- EMBEDDINGS_BASE_URL
- EMBEDDINGS_MODEL (nomic-embed-text-v1.5)
"""

import os
import argparse
import requests
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "county_chunks"

API_KEY = os.getenv("EMBEDDINGS_API_KEY", "").strip()
BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "").strip()
MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text-v1.5").strip()


def embed_query(text: str):
    url = BASE_URL.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "input": [text]}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Embeddings error {r.status_code}: {r.text}")
    return r.json()["data"][0]["embedding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Your search query in plain English")
    parser.add_argument("--k", type=int, default=5, help="Top K results")
    parser.add_argument("--county", type=str, default="", help="Optional filter: exact county folder name")
    args = parser.parse_args()

    if not API_KEY or not BASE_URL:
        raise RuntimeError("Missing EMBEDDINGS_API_KEY or EMBEDDINGS_BASE_URL in .env")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(COLLECTION_NAME)

    q_emb = embed_query(args.query)

    where = {"county": args.county} if args.county else None

    res = col.query(
        query_embeddings=[q_emb],
        n_results=args.k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    print("\nQUERY:", args.query)
    if args.county:
        print("FILTER county:", args.county)
    print("\nTop results:\n")

    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        doc = res["documents"][0][i]
        dist = res["distances"][0][i]

        county = meta.get("county")
        pdf = meta.get("pdf_file")
        ps = meta.get("page_start")
        pe = meta.get("page_end")

        snippet = doc[:350].replace("\n", " ")
        print(f"{i+1}) distance={dist:.4f}")
        print(f"   County: {county}")
        print(f"   Source: {pdf} (pages {ps}-{pe})")
        print(f"   Snippet: {snippet}")
        print("-" * 90)


if __name__ == "__main__":
    main()