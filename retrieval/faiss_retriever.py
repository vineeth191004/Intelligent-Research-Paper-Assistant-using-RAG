import os
import json
import faiss
import numpy as np
from config import settings
from embeddings.embedder import DocumentEmbedder

class FAISSRetriever:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.embedder = DocumentEmbedder()

    def fit(self, chunks: list[dict]):
        """
        Builds FAISS index on text chunks.
        Each chunk is a dictionary with 'id', 'text', and 'metadata'.
        """
        self.chunks = chunks
        if not chunks:
            self.index = None
            return

        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        
        # Fetch dimensions
        dimension = embeddings.shape[1]
        
        # IndexFlatIP uses Inner Product (equivalent to Cosine Similarity when vectors are normalized)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def save(self, index_path: str = None, meta_path: str = None) -> bool:
        """
        Saves FAISS index and chunk metadata to storage.
        """
        if self.index is None:
            return False
            
        if index_path is None:
            index_path = str(settings.FAISS_INDEX_PATH)
        if meta_path is None:
            meta_path = str(settings.CHUNKS_METADATA_PATH)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            
        return True

    def load(self, index_path: str = None, meta_path: str = None) -> bool:
        """
        Loads FAISS index and chunk metadata from storage.
        """
        if index_path is None:
            index_path = str(settings.FAISS_INDEX_PATH)
        if meta_path is None:
            meta_path = str(settings.CHUNKS_METADATA_PATH)

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        self.index = faiss.read_index(index_path)
        
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        return True

    def search(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> list[dict]:
        """
        Queries the FAISS index.
        Returns matching chunks with dense similarity scores.
        """
        if self.index is None or not self.chunks:
            return []

        query_vector = self.embedder.embed_query(query)
        # Reshape to (1, dimension) for FAISS compatibility
        query_vector = np.expand_dims(query_vector, axis=0)

        # Search index
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["dense_score"] = float(score)
            chunk["dense_rank"] = rank
            results.append(chunk)

        return results
