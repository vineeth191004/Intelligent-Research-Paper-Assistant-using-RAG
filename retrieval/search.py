import os
from sentence_transformers import CrossEncoder
from config import settings
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever

class HybridSearchEngine:
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.faiss_retriever = FAISSRetriever()
        self.reranker = None  # Loaded lazily on first search query to conserve memory on startup

    def load_indices(self) -> bool:
        """
        Loads the indices from local files.
        Returns True if successful, False otherwise.
        """
        bm25_success = self.bm25_retriever.load()
        faiss_success = self.faiss_retriever.load()
        return bm25_success and faiss_success

    def index_documents(self, chunks: list[dict]):
        """
        Indexes chunks into both BM25 and FAISS indexes and saves them to disk.
        """
        if not chunks:
            return

        # Fit and save sparse index
        self.bm25_retriever.fit(chunks)
        self.bm25_retriever.save()

        # Fit and save dense index
        self.faiss_retriever.fit(chunks)
        self.faiss_retriever.save()

    def _get_reranker(self) -> CrossEncoder:
        """
        Returns (and lazily loads) the Cross-Encoder model.
        """
        if self.reranker is None:
            print(f"Loading reranking model: {settings.RERANK_MODEL_NAME} on CPU...")
            self.reranker = CrossEncoder(settings.RERANK_MODEL_NAME, device="cpu")
            print("Reranking model loaded successfully.")
        return self.reranker

    def search(
        self, 
        query: str, 
        top_k_retrieve: int = settings.TOP_K_RETRIEVAL, 
        top_k_rerank: int = settings.TOP_K_RERANK
    ) -> list[dict]:
        """
        Performs hybrid retrieval:
        1. Queries dense FAISS index.
        2. Queries sparse BM25 index.
        3. Fuses findings using Reciprocal Rank Fusion (RRF).
        4. Reranks final results using Cross-Encoder.
        """
        # Retrieve candidate lists from both indices
        dense_results = self.faiss_retriever.search(query, top_k=top_k_retrieve)
        sparse_results = self.bm25_retriever.search(query, top_k=top_k_retrieve)

        if not dense_results and not sparse_results:
            return []

        # Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, k=settings.RRF_K)

        # Cross-Encoder Reranking
        reranker = self._get_reranker()
        
        # Prepare text pairs for Cross-Encoder evaluation
        pairs = [[query, doc["text"]] for doc in fused_results]
        
        # Predict relevance scores (larger is more relevant)
        scores = reranker.predict(pairs)

        for doc, score in zip(fused_results, scores):
            doc["rerank_score"] = float(score)

        # Sort based on rerank scores descending
        reranked_results = sorted(fused_results, key=lambda x: x["rerank_score"], reverse=True)

        return reranked_results[:top_k_rerank]

    def _reciprocal_rank_fusion(self, dense_results: list[dict], sparse_results: list[dict], k: int = 60) -> list[dict]:
        """
        Merges results from dense and sparse search using Reciprocal Rank Fusion.
        Formula: RRF_score(d) = sum_{m in retrievers} (1 / (k + rank_m(d)))
        """
        scores = {}

        # 1. Process Dense Results
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["id"]
            if doc_id not in scores:
                scores[doc_id] = {
                    "doc": doc,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "rrf_score": 0.0
                }
            scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

        # 2. Process Sparse Results
        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc["id"]
            if doc_id not in scores:
                scores[doc_id] = {
                    "doc": doc,
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "rrf_score": 0.0
                }
            else:
                # Merge dictionary updates (keeps score fields from both retrievers)
                scores[doc_id]["doc"].update(doc)
                scores[doc_id]["sparse_rank"] = rank

            scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

        # 3. Assemble and Sort by fusion score descending
        fused = []
        for doc_id, item in scores.items():
            doc = item["doc"]
            doc["rrf_score"] = item["rrf_score"]
            doc["dense_rank"] = item["dense_rank"]
            doc["sparse_rank"] = item["sparse_rank"]
            fused.append(doc)

        return sorted(fused, key=lambda x: x["rrf_score"], reverse=True)
