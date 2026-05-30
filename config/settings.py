import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Storage Paths
PDF_UPLOAD_DIR = DATA_DIR / "uploads"
PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

INDEX_DIR = DATA_DIR / "indices"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "faiss_index"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
CHUNKS_METADATA_PATH = INDEX_DIR / "chunks_metadata.json"

# Ingestion Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Models Settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Search Parameters
TOP_K_RETRIEVAL = 15      # Retrieve 15 from each of BM25 and Dense
TOP_K_RERANK = 2          # (Optimized for speed) Keep only top 2 chunks for the LLM
RRF_K = 60                # Reciprocal Rank Fusion constant
# Ollama Local LLM Setup
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = "llama3.2:1b"  # 1-Billion parameter model (much faster on CPU)

def is_ollama_running():
    import requests
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
