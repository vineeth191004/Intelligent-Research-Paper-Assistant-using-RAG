import json
import requests
from config import settings

class OllamaEvaluator:
    def __init__(self):
        """Initializes the RAG evaluator using local Ollama."""
        self.base_url = settings.OLLAMA_BASE_URL
        self.model_name = settings.OLLAMA_MODEL_NAME

    def _call_ollama_json(self, prompt: str) -> str:
        """Calls Ollama in JSON mode."""
        if not settings.is_ollama_running():
            return "{}"

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "{}")
        except requests.exceptions.RequestException as e:
            print(f"Ollama Eval Error: {e}")
            return "{}"

    def _get_llm_evaluation(self, prompt: str) -> dict:
        """Submits the prompt to Ollama and parses the JSON score/reason response."""
        text = self._call_ollama_json(prompt)
        try:
            data = json.loads(text.strip())
            return {
                "score": float(data.get("score", 0.0)),
                "reason": str(data.get("reason", "No reason provided."))
            }
        except json.JSONDecodeError:
            # Fallback for models that ignore JSON mode
            return {"score": 0.0, "reason": "Failed to parse Ollama evaluation output as JSON."}

    def evaluate_faithfulness(self, contexts: list[str], answer: str) -> dict:
        """
        Evaluates if the answer is grounded in the retrieved contexts.
        Returns a dict: {"score": 0.0-1.0, "reason": "str"}
        """
        context_str = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
        prompt = f"""You are an expert evaluator. Evaluate the faithfulness of the answer based on the provided context.
An answer is faithful if all claims made in the answer can be inferred from the context.
Give a score between 0.0 (completely unfaithful/hallucinated) and 1.0 (completely faithful).

Contexts:
{context_str}

Answer:
{answer}

Output a JSON object with exactly two keys: "score" (a float between 0 and 1) and "reason" (a short string explaining the score).
Example: {{"score": 1.0, "reason": "The answer directly summarizes the provided context."}}
"""
        return self._get_llm_evaluation(prompt)

    def evaluate_context_relevance(self, query: str, contexts: list[str]) -> dict:
        """
        Evaluates if the retrieved contexts are relevant to the query.
        Returns a dict: {"score": 0.0-1.0, "reason": "str"}
        """
        context_str = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
        prompt = f"""You are an expert evaluator. Evaluate the relevance of the retrieved contexts to the given user query.
Give a score between 0.0 (completely irrelevant) and 1.0 (highly relevant and sufficient to answer the query).

Query: {query}

Contexts:
{context_str}

Output a JSON object with exactly two keys: "score" (a float between 0 and 1) and "reason" (a short string explaining the score).
Example: {{"score": 0.8, "reason": "Context 1 and 2 provide direct information about the query."}}
"""
        return self._get_llm_evaluation(prompt)

    def evaluate_answer_relevance(self, query: str, answer: str) -> dict:
        """
        Evaluates if the final answer directly addresses the user's query.
        Returns a dict: {"score": 0.0-1.0, "reason": "str"}
        """
        prompt = f"""You are an expert evaluator. Evaluate the relevance of the answer to the user's query.
Does the answer directly address the question? Does it avoid irrelevant tangents?
Give a score between 0.0 (completely irrelevant) and 1.0 (highly relevant).

Query: {query}
Answer: {answer}

Output a JSON object with exactly two keys: "score" (a float between 0 and 1) and "reason" (a short string explaining the score).
Example: {{"score": 0.9, "reason": "The answer directly addresses the query but includes a minor unrelated detail."}}
"""
        return self._get_llm_evaluation(prompt)

    def evaluate_retrieval(self, retrieved_docs: list[dict], ground_truth_pages: list[int], k: int) -> dict:
        """
        Evaluates retrieval performance using Precision@K and Recall@K.
        """
        top_k_docs = retrieved_docs[:k]
        retrieved_pages = [doc["metadata"]["page"] for doc in top_k_docs]
        
        hits = set(retrieved_pages).intersection(set(ground_truth_pages))
        
        precision = len(hits) / k if k > 0 else 0.0
        recall = len(hits) / len(ground_truth_pages) if len(ground_truth_pages) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "hits": list(hits)
        }
