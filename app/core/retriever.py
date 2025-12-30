from app.core.embeddings import EmbeddingModel
from app.core.vector_store import VectorStore
from app.core.hybrid_search import HybridSearch
from app.config import TOP_K

class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.hybrid_search = HybridSearch(vector_store)

    def retrieve(self, query: str):
        """Retrieve documents using hybrid search (BM25 + Vector + Rerank)."""
        return self.hybrid_search.search(query, TOP_K)
