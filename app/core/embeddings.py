from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import EMBEDDING_MODEL

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=False))

    def embed_query(self, query: str) -> np.ndarray:
        return np.array(self.model.encode([query]))[0]
