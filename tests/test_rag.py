import sys
import unittest
from pathlib import Path

# Add project root directory to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.parser import clean_text
from retrieval.bm25_retriever import BM25Retriever
from retrieval.search import HybridSearchEngine
from evaluation.metrics import RAGEvaluator

class TestRAGComponents(unittest.TestCase):
    
    def test_clean_text(self):
        """
        Tests cleaning of raw text to remove redundant whitespace and unreadable characters.
        """
        text = "Hello \n  World! \t This is a test \x00."
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "Hello World! This is a test .")
        
    def test_bm25_retriever(self):
        """
        Tests fitting and searching BM25 sparse index.
        """
        chunks = [
            {
                "id": "doc1_p1_c0", 
                "text": "The quick brown fox jumps over the lazy dog", 
                "metadata": {"source": "doc1.pdf", "page": 1, "chunk_idx": 0}
            },
            {
                "id": "doc2_p1_c0", 
                "text": "Deep learning models require large datasets", 
                "metadata": {"source": "doc2.pdf", "page": 1, "chunk_idx": 0}
            },
            {
                "id": "doc3_p1_c0", 
                "text": "This is a completely random sentence to increase corpus size", 
                "metadata": {"source": "doc3.pdf", "page": 1, "chunk_idx": 0}
            }
        ]
        
        retriever = BM25Retriever()
        retriever.fit(chunks)
        
        # Query matching document 2
        results = retriever.search("deep learning", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc2_p1_c0")
        self.assertTrue("sparse_score" in results[0])
        
    def test_rrf_logic(self):
        """
        Tests the Reciprocal Rank Fusion algorithm combining two ranked lists.
        """
        dense_results = [
            {"id": "doc1", "text": "text1", "metadata": {"source": "doc1.pdf", "page": 1}},
            {"id": "doc2", "text": "text2", "metadata": {"source": "doc2.pdf", "page": 1}}
        ]
        sparse_results = [
            {"id": "doc2", "text": "text2", "metadata": {"source": "doc2.pdf", "page": 1}},
            {"id": "doc3", "text": "text3", "metadata": {"source": "doc3.pdf", "page": 1}}
        ]
        
        engine = HybridSearchEngine()
        fused = engine._reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        
        # doc2 is present in both lists and should bubble to the top
        self.assertEqual(fused[0]["id"], "doc2")
        self.assertEqual(len(fused), 3)
        self.assertTrue("rrf_score" in fused[0])
        
    def test_retrieval_evaluation(self):
        """
        Tests calculation of Precision@K and Recall@K using page numbers.
        """
        evaluator = RAGEvaluator()
        retrieved_docs = [
            {"id": "doc1", "metadata": {"page": 1}},
            {"id": "doc2", "metadata": {"page": 2}},
            {"id": "doc3", "metadata": {"page": 3}}
        ]
        ground_truth_pages = [2, 4]
        
        # Test Precision/Recall with top-2 candidates
        metrics = evaluator.evaluate_retrieval(retrieved_docs, ground_truth_pages, k=2)
        
        # Retrieved pages top-2 are 1, 2. Ground truth pages are 2, 4.
        # Intersection is [2] (size 1)
        # Precision = 1 / 2 = 0.5
        # Recall = 1 / 2 = 0.5
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["hits"], [2])

if __name__ == "__main__":
    unittest.main()
