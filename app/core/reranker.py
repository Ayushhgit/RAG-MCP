from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from app.config import RERANKING_MODEL
from app.utils.logger import logger

class Reranker:
    def __init__(self):
        try:
            self.model = CrossEncoder(RERANKING_MODEL)
            logger.info(f"Initialized reranker with model: {RERANKING_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}. Using fallback.")
            self.model = None

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query."""
        if not self.model or not documents:
            return documents[:top_k]

        try:
            # Prepare input for cross-encoder
            pairs = [[query, doc.get("text", "")] for doc in documents]

            # Get relevance scores
            scores = self.model.predict(pairs)

            # Sort documents by score (higher is better for cross-encoder)
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = [doc for doc, score in scored_docs[:top_k]]

            logger.info(f"Reranked {len(documents)} documents, selected top {len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"Error during reranking: {e}. Returning original documents.")
            return documents[:top_k]