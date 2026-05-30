import json
import requests
from config import settings

class OllamaGenerator:
    def __init__(self):
        """Initializes the Ollama Generator."""
        self.base_url = settings.OLLAMA_BASE_URL
        self.model_name = settings.OLLAMA_MODEL_NAME

    def _call_ollama(self, prompt: str, json_mode: bool = False) -> str:
        """Calls the Ollama API with the given prompt."""
        if not settings.is_ollama_running():
            return "⚠️ **Ollama is not running.** Please start the Ollama service on your machine (e.g. run `ollama serve` or start the Ollama app)."

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Ollama API Error: {e}")
            return f"⚠️ **Error communicating with Ollama:** {str(e)}"

    def condense_query(self, chat_history: list[dict], query: str) -> str:
        """
        Condenses conversational follow-ups into standalone search queries.
        """
        if not chat_history:
            return query

        history_str = ""
        for turn in chat_history:
            history_str += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"

        prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone search query that contains all necessary context from the conversation.
Do NOT answer the question. Just output the standalone search query.
If the follow-up question is already standalone, output it exactly.

Conversation History:
{history_str}
Follow-up Question: {query}

Standalone Query:"""

        text = self._call_ollama(prompt)
        if text.startswith("⚠️"):
            return query # Fallback to original query if Ollama is down

        condensed = text.strip()
        if condensed.lower().startswith("standalone query:"):
            condensed = condensed[len("standalone query:"):].strip()
        
        return condensed if condensed else query

    def generate_answer(self, query: str, contexts: list[dict], chat_history: list[dict]) -> dict:
        """
        Generates a grounded, citation-backed answer using Ollama.
        """
        context_str = ""
        for idx, doc in enumerate(contexts, start=1):
            source = doc['metadata']['source']
            page = doc['metadata']['page']
            context_str += f"[{idx}] (Source: {source}, Page: {page})\nText: {doc['text']}\n\n"

        history_str = ""
        for turn in chat_history[-5:]:
            history_str += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"

        prompt = f"""You are an advanced AI Research Paper Assistant. Answer the user's question using ONLY the retrieved context passages below.
For every claim you make, cite the context index in brackets, e.g. [1] or [2].
Structure your answer with bullet points or paragraphs where appropriate.
If the answer cannot be found in the context, clearly state that.

Retrieved Context Passages:
{context_str}

Conversation History:
{history_str}

User Question: {query}
Answer:"""

        answer_text = self._call_ollama(prompt)

        citations = []
        if not answer_text.startswith("⚠️"):
            for doc in contexts:
                citations.append({
                    "source": doc["metadata"]["source"],
                    "page": doc["metadata"]["page"],
                    "text": doc["text"],
                    "dense_score": doc.get("dense_score"),
                    "sparse_score": doc.get("sparse_score"),
                    "rrf_score": doc.get("rrf_score"),
                    "rerank_score": doc.get("rerank_score", 0.0)
                })

        return {"answer": answer_text, "citations": citations}
