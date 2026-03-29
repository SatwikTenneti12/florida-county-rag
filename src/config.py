"""
Centralized configuration for the Florida County RAG system.

All paths, API settings, and model parameters are managed here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"
CHROMA_DIR = ROOT / "chroma_db"

# Processed data files
PAGES_PATH = PROCESSED_DIR / "pages.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
PREDICTIONS_PATH = PROCESSED_DIR / "topic_evidence_by_county.csv"
EVAL_PATH = PROCESSED_DIR / "eval_joined.csv"
LABELS_PATH = MANIFESTS_DIR / "county_labels.csv"

# Chroma settings
COLLECTION_NAME = "county_chunks"

# API settings
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY", "").strip()
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", "").strip()
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text-v1.5").strip()

# LLM settings (for RAG generation)
LLM_API_KEY = os.getenv("LLM_API_KEY", EMBEDDINGS_API_KEY).strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", EMBEDDINGS_BASE_URL).strip()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instruct").strip()

# Chunking parameters
CHUNK_SIZE = 4000  # characters (~600-800 tokens)
CHUNK_OVERLAP = 800  # overlap for context continuity
MIN_CHUNK_SIZE = 200  # minimum chunk size to keep

# Retrieval parameters
DEFAULT_TOP_K = 10
RERANK_TOP_K = 5

# Classification topics
TOPICS = {
    "wildlife_corridors": {
        "display_name": "Wildlife Corridors",
        "description": "Policies establishing or protecting wildlife corridors for animal movement",
        "keywords": ["wildlife corridor", "corridor linkage", "habitat corridor", 
                     "ecological corridor", "greenway corridor", "movement corridor"],
        "query": "wildlife corridor policies for habitat connectivity and animal movement"
    },
    "wildlife_crossings": {
        "display_name": "Wildlife Crossings",
        "description": "Infrastructure policies for wildlife underpasses, overpasses, or crossing structures",
        "keywords": ["wildlife crossing", "wildlife underpass", "wildlife overpass", 
                     "animal crossing", "ecopassage", "wildlife bridge"],
        "query": "wildlife crossing structures underpasses overpasses for road safety"
    },
    "land_acquisition": {
        "display_name": "Land Acquisition",
        "description": "Policies for acquiring land for conservation purposes",
        "keywords": ["land acquisition", "acquire land", "conservation acquisition",
                     "purchase land", "conservation easement", "fee simple acquisition"],
        "query": "land acquisition policies for conservation and environmental protection"
    },
    "wildlife_surveys": {
        "display_name": "Wildlife Surveys",
        "description": "Requirements for wildlife surveys before development",
        "keywords": [
            "wildlife survey",
            "species survey",
            "habitat survey",
            "biological survey",
            "baseline survey",
            "preconstruction survey",
            "wildlife inventory",
            "species inventory",
            "wildlife monitoring",
            "species monitoring",
        ],
        "query": "policies that require wildlife/species/habitat surveys, inventories, or monitoring (baseline or preconstruction), as a condition of development or land-use change"
    },
    "open_space": {
        "display_name": "Open Space Planning",
        "description": "Policies for preserving open space, greenways, and conservation areas",
        "keywords": ["open space", "greenway", "green space", "conservation area",
                     "natural area", "preservation area", "open lands"],
        "query": "open space preservation greenway conservation area planning policies"
    }
}

# County name normalization mapping
COUNTY_ALIASES = {
    # Common variations in folder names -> canonical name
    "miami-dade": "Miami-Dade County",
    "st. johns": "St. Johns County", 
    "st johns": "St. Johns County",
    "st. lucie": "St. Lucie County",
    "st lucie": "St. Lucie County",
}


def validate_config():
    """Check that required configuration is present."""
    errors = []
    
    if not EMBEDDINGS_API_KEY:
        errors.append("Missing EMBEDDINGS_API_KEY in .env")
    if not EMBEDDINGS_BASE_URL:
        errors.append("Missing EMBEDDINGS_BASE_URL in .env")
    
    if errors:
        raise RuntimeError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True
