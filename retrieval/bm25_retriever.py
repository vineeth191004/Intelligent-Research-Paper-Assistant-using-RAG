import pickle
import re
import os
from rank_bm25 import BM25Okapi
from config import settings

def tokenize(text: str) -> list[str]:
    """
    Splits text into lowercase alphanumeric tokens.
    """
    return re.findall(r'\w+', text.lower())

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def fit(self, chunks: list[dict]):
        """
        Fits BM25 index on text chunks.
        Each chunk is a dictionary with 'id', 'text', and 'metadata'.
        """
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return
            
        tokenized_corpus = [tokenize(chunk['text']) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, filepath: str = None) -> bool:
        """
        Pickles the BM25 model and raw chunks.
        """
        if self.bm25 is None:
            return False
        if filepath is None:
            filepath = str(settings.BM25_INDEX_PATH)
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump((self.bm25, self.chunks), f)
        return True

    def load(self, filepath: str = None) -> bool:
        """
        Loads the pickled BM25 model and chunks.
        """
        if filepath is None:
            filepath = str(settings.BM25_INDEX_PATH)
            
        if not os.path.exists(filepath):
            return False
            
        with open(filepath, 'rb') as f:
            self.bm25, self.chunks = pickle.load(f)
        return True

    def search(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> list[dict]:
        """
        Searches the BM25 index for the query.
        Returns a list of dictionaries with matching chunks and BM25 scores.
        """
        if self.bm25 is None or not self.chunks:
            return []
            
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort documents by score descending
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for rank, (idx, score) in enumerate(scored_indices, start=1):
            chunk = self.chunks[idx].copy()
            chunk["sparse_score"] = float(score)
            chunk["sparse_rank"] = rank
            results.append(chunk)
            
        return results
