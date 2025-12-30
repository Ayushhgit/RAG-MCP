from app.core.vector_store import VectorStore
from app.core.embeddings import EmbeddingModel
from app.config import TOP_K
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.utils.logger import logger

def search_knowledge(query: str, top_k: int = TOP_K) -> SearchResponse:
    """Search the knowledge base for relevant documents."""
    try:
        logger.info(f"Searching for: {query}")
        embedder = EmbeddingModel()
        vector = embedder.embed_query(query)

        store = VectorStore(len(vector))
        raw_results = store.search(vector, top_k)

        results = [
            SearchResult(
                text=result.get("text", ""),
                source=result.get("source", ""),
                score=0.0  # FAISS doesn't return scores in this implementation
            )
            for result in raw_results
        ]

        logger.info(f"Found {len(results)} results")
        return SearchResponse(results=results, query=query)
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        raise
