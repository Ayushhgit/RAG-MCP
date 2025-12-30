from app.core.chunking import chunk_text
from app.core.embeddings import EmbeddingModel
from app.core.vector_store import VectorStore
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.schemas.ingest import IngestRequest, IngestResponse
from app.utils.logger import logger

def ingest_documents(texts: list[str]) -> IngestResponse:
    """Ingest documents into the vector store."""
    try:
        logger.info(f"Ingesting {len(texts)} documents")
        embedder = EmbeddingModel()
        chunks = []
        metadatas = []

        for i, text in enumerate(texts):
            text_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk in text_chunks:
                chunks.append(chunk)
                metadatas.append({"source": f"doc_{i}", "text": chunk})

        if chunks:
            vectors = embedder.embed_documents(chunks)
            store = VectorStore(vectors.shape[1])
            store.add(vectors, metadatas)
            logger.info(f"Successfully ingested {len(chunks)} chunks from {len(texts)} documents")

        return IngestResponse(
            documents=len(texts),
            chunks=len(chunks)
        )
    except Exception as e:
        logger.error(f"Error ingesting documents: {str(e)}")
        raise
