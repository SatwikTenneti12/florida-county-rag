"""
Vector retriever for the RAG system.

Provides semantic search over the Chroma vector database with citation metadata.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, CHUNKS_PATH, EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL
from utils.embeddings import embed_query
from utils.county_normalizer import normalize_county_name


@dataclass
class RetrievedChunk:
    """A retrieved chunk with metadata for citation."""
    text: str
    county: str
    pdf_file: str
    page_start: int
    page_end: int
    distance: float
    chunk_id: str
    
    @property
    def citation(self) -> str:
        """Generate a formatted citation string."""
        return f"{self.county}, {self.pdf_file}, pp. {self.page_start}-{self.page_end}"
    
    @property
    def short_citation(self) -> str:
        """Short citation for inline use."""
        return f"[{self.county}, p.{self.page_start}]"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "county": self.county,
            "pdf_file": self.pdf_file,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "distance": self.distance,
            "citation": self.citation
        }


class VectorRetriever:
    """Retriever for semantic search over county comprehensive plans."""
    
    def __init__(self, chroma_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self._bootstrap_collection(collection_name)

    def _bootstrap_collection(self, collection_name: str):
        """
        Build a collection on first run when Space storage is empty.
        Uses `data/processed/chunks.jsonl` and embedding API secrets.
        """
        if not CHUNKS_PATH.exists():
            raise RuntimeError(
                f"Collection [{collection_name}] does not exist and missing chunks file at {CHUNKS_PATH}. "
                "Add data/processed/chunks.jsonl to the repo."
            )

        if not EMBEDDINGS_BASE_URL or not EMBEDDINGS_API_KEY:
            raise RuntimeError(
                "Collection missing and embeddings config is incomplete. "
                "Set EMBEDDINGS_BASE_URL and EMBEDDINGS_API_KEY in environment."
            )

        from indexing.build_chroma import load_chunks, stable_id, embed_texts

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if collection.count() > 0:
            return collection

        limit = int(os.getenv("BOOTSTRAP_CHUNK_LIMIT", "0") or "0")
        batch_size = int(os.getenv("BOOTSTRAP_EMBED_BATCH", "32") or "32")
        chunks = load_chunks(limit=limit if limit > 0 else None)

        unique = {}
        for r in chunks:
            cid = stable_id(
                r["county"],
                r["pdf_file"],
                int(r["page_start"]),
                int(r["page_end"]),
                r["text"],
            )
            unique[cid] = r

        items = list(unique.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
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
                base_url=EMBEDDINGS_BASE_URL,
                api_key=EMBEDDINGS_API_KEY,
                model=EMBEDDINGS_MODEL,
            )
            collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )

        return collection
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        county_filter: Optional[str] = None,
        distance_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            county_filter: Optional county name to filter by
            distance_threshold: Optional max distance to include (lower = more similar)
            
        Returns:
            List of RetrievedChunk objects sorted by relevance
        """
        query_embedding = embed_query(query)
        
        where_filter = None
        if county_filter:
            where_filter = {"county": county_filter}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )
        
        chunks = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            if distance_threshold and distance > distance_threshold:
                continue
            
            chunk = RetrievedChunk(
                text=results["documents"][0][i],
                county=meta.get("county", "Unknown"),
                pdf_file=meta.get("pdf_file", "Unknown"),
                page_start=meta.get("page_start", 0),
                page_end=meta.get("page_end", 0),
                distance=distance,
                chunk_id=results["ids"][0][i]
            )
            chunks.append(chunk)
        
        return chunks
    
    def retrieve_for_county(
        self,
        query: str,
        county: str,
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Retrieve chunks for a specific county."""
        return self.retrieve(query, top_k=top_k, county_filter=county)
    
    def retrieve_multi_query(
        self,
        queries: List[str],
        top_k_per_query: int = 5,
        county_filter: Optional[str] = None,
        deduplicate: bool = True
    ) -> List[RetrievedChunk]:
        """
        Retrieve using multiple query variations and merge results.
        
        Useful for improving recall by searching with different phrasings.
        """
        all_chunks = {}
        
        for query in queries:
            chunks = self.retrieve(query, top_k=top_k_per_query, county_filter=county_filter)
            for chunk in chunks:
                if deduplicate:
                    if chunk.chunk_id not in all_chunks:
                        all_chunks[chunk.chunk_id] = chunk
                    elif chunk.distance < all_chunks[chunk.chunk_id].distance:
                        all_chunks[chunk.chunk_id] = chunk
                else:
                    all_chunks[f"{chunk.chunk_id}_{query}"] = chunk
        
        # Sort by distance
        sorted_chunks = sorted(all_chunks.values(), key=lambda c: c.distance)
        return sorted_chunks
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }


# Singleton retriever
_retriever = None

def get_retriever() -> VectorRetriever:
    """Get or create the singleton retriever."""
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever()
    return _retriever
