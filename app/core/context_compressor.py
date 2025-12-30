from typing import List, Dict, Any, Optional
from app.core.llm import generate_answer
from app.core.router import AgentType
from app.config import CONTEXT_MAX_LENGTH, CONTEXT_COMPRESSION_RATIO
from app.utils.logger import logger

class ContextCompressor:
    def __init__(self, max_context_length: int = CONTEXT_MAX_LENGTH, compression_ratio: float = CONTEXT_COMPRESSION_RATIO):
        self.max_context_length = max_context_length
        self.compression_ratio = compression_ratio

    def compress_context(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Compress retrieved documents to fit within context limits."""
        if not documents:
            return ""

        # Calculate total length
        total_length = sum(len(doc.get('text', '')) for doc in documents)

        # If within limits, return as-is
        if total_length <= self.max_context_length:
            return self._format_documents(documents)

        # Need compression
        logger.info(f"Compressing context: {total_length} chars -> ~{int(total_length * self.compression_ratio)}")

        # Method 1: Extractive compression (select most relevant parts)
        compressed_docs = self._extractive_compression(query, documents)

        # If still too long, use abstractive compression
        compressed_text = self._format_documents(compressed_docs)
        if len(compressed_text) > self.max_context_length:
            compressed_text = self._abstractive_compression(query, compressed_docs)

        return compressed_text

    def _extractive_compression(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract most relevant sentences/sections from documents."""
        compressed_docs = []

        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue

            # Simple sentence-based extraction
            sentences = self._split_into_sentences(text)

            # Score sentences by relevance to query
            query_words = set(query.lower().split())
            scored_sentences = []

            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                # Simple overlap score
                score = len(query_words.intersection(sentence_words))
                scored_sentences.append((sentence, score))

            # Select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in scored_sentences[:3]]  # Top 3 sentences

            compressed_text = ' '.join(top_sentences)
            if len(compressed_text) > 100:  # Minimum length threshold
                compressed_docs.append({
                    'text': compressed_text,
                    'source': doc.get('source', 'compressed'),
                    'compressed': True
                })

        return compressed_docs

    def _abstractive_compression(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Use LLM to generate compressed summary of documents."""
        try:
            # Combine all document texts
            all_text = self._format_documents(documents)

            # Create compression prompt
            compression_prompt = f"""
            Given the query: "{query}"

            Summarize the following context in a concise way that answers the query.
            Keep only the most relevant information. Be brief but comprehensive.

            Context:
            {all_text}

            Summary:"""

            # Use a lightweight model for compression
            compressed = generate_answer("", compression_prompt, AgentType.GENERAL_QA)

            logger.info(f"Abstractive compression: {len(all_text)} -> {len(compressed)} chars")
            return compressed

        except Exception as e:
            logger.error(f"Abstractive compression failed: {e}")
            # Fallback to extractive
            return self._format_documents(self._extractive_compression(query, documents))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into context string."""
        formatted_parts = []
        for doc in documents:
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')
            formatted_parts.append(f"Source: {source}\n{text}")

        return "\n\n".join(formatted_parts)