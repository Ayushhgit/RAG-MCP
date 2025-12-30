from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any, Tuple
from app.core.embeddings import EmbeddingModel
from app.core.vector_store import VectorStore
from app.core.reranker import Reranker
from app.config import TOP_K, RERANK_TOP_K
from app.utils.logger import logger

class HybridSearch:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.embedder = EmbeddingModel()
        self.reranker = Reranker()
        self.bm25 = None
        self.documents = []
        self._load_bm25_index()

    def _load_bm25_index(self):
        """Load or create BM25 index from stored documents."""
        try:
            # Get all documents from vector store metadata
            if hasattr(self.vector_store, 'metadata') and self.vector_store.metadata:
                self.documents = [doc.get('text', '') for doc in self.vector_store.metadata]
                if self.documents:
                    # Tokenize documents for BM25
                    tokenized_docs = [doc.lower().split() for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized_docs)
                    logger.info(f"Initialized BM25 index with {len(self.documents)} documents")
        except Exception as e:
            logger.warning(f"Failed to initialize BM25 index: {e}")
            self.bm25 = None

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform BM25 search and return (doc_index, score) pairs."""
        if not self.bm25 or not self.documents:
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k results with scores
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform vector search and return (doc_index, score) pairs."""
        try:
            query_vector = self.embedder.embed_query(query)
            # Note: FAISS doesn't return meaningful scores, so we'll use dummy scores
            results = self.vector_store.search(query_vector, top_k)
            return [(i, 1.0 - (i * 0.1)) for i in range(len(results))]  # Decreasing dummy scores
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _combine_scores(self, bm25_results: List[Tuple[int, float]],
                       vector_results: List[Tuple[int, float]],
                       alpha: float = 0.5) -> List[Tuple[int, float]]:
        """Combine BM25 and vector search scores using weighted combination."""
        # Create score dictionaries
        bm25_scores = dict(bm25_results)
        vector_scores = dict(vector_results)

        # Normalize scores
        if bm25_scores:
            bm25_max = max(bm25_scores.values())
            bm25_min = min(bm25_scores.values())
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
            bm25_scores = {idx: (score - bm25_min) / bm25_range for idx, score in bm25_scores.items()}

        if vector_scores:
            vector_max = max(vector_scores.values())
            vector_min = min(vector_scores.values())
            vector_range = vector_max - vector_min if vector_max != vector_min else 1
            vector_scores = {idx: (score - vector_min) / vector_range for idx, score in vector_scores.items()}

        # Combine scores
        all_indices = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined_scores = {}

        for idx in all_indices:
            bm25_score = bm25_scores.get(idx, 0.0)
            vector_score = vector_scores.get(idx, 0.0)
            combined_scores[idx] = alpha * bm25_score + (1 - alpha) * vector_score

        # Sort by combined score
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Perform hybrid search combining BM25, vector search, and reranking."""
        try:
            # Get more candidates for better reranking
            search_top_k = max(top_k * 3, 15)

            # Perform BM25 and vector search
            bm25_results = self._bm25_search(query, search_top_k)
            vector_results = self._vector_search(query, search_top_k)

            # Combine scores
            combined_results = self._combine_scores(bm25_results, vector_results)

            # Get documents for reranking
            candidate_docs = []
            for idx, score in combined_results[:search_top_k]:
                if idx < len(self.vector_store.metadata):
                    doc = self.vector_store.metadata[idx].copy()
                    doc['hybrid_score'] = score
                    candidate_docs.append(doc)

            # Rerank the combined results
            reranked_docs = self.reranker.rerank(query, candidate_docs, top_k)

            logger.info(f"Hybrid search found {len(reranked_docs)} results for query: {query[:50]}...")
            return reranked_docs

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector search only
            try:
                query_vector = self.embedder.embed_query(query)
                results = self.vector_store.search(query_vector, top_k)
                return results
            except Exception as e2:
                logger.error(f"Fallback search also failed: {e2}")
                return []