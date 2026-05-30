import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings

class DocumentEmbedder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DocumentEmbedder, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        if self._model is None:
            print(f"Loading embedding model: {model_name} on CPU...")
            # Forces loading on CPU to satisfy cheap hosting / no GPU constraints
            self._model = SentenceTransformer(model_name, device="cpu")
            print("Embedding model loaded successfully.")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embeds a list of documents/chunks and returns a numpy array.
        """
        if not texts:
            return np.empty((0, self.get_embedding_dimension()), dtype=np.float32)
        
        # encode returns numpy arrays directly
        embeddings = self._model.encode(
            texts, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True  # normalize to unit vectors for cosine similarity
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query text.
        """
        embedding = self._model.encode(
            query, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)

    def get_embedding_dimension(self) -> int:
        """
        Returns the embedding dimension size.
        """
        return self._model.get_sentence_embedding_dimension()
