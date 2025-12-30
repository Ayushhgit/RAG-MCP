from app.core.embeddings import EmbeddingModel
from app.core.vector_store import VectorStore
from app.config import TOP_K

class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.embedder = EmbeddingModel()
        self.vector_store = vector_store

    def retrieve(self, query: str):
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.search(query_vector, TOP_K)
