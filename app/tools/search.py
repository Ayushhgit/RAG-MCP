from app.core.vector_store import VectorStore
from app.core.embeddings import EmbeddingModel
from app.config import TOP_K

def search_knowledge(query: str):
    embedder = EmbeddingModel()
    vector = embedder.embed_query(query)

    store = VectorStore(len(vector))
    results = store.search(vector, TOP_K)

    return results
