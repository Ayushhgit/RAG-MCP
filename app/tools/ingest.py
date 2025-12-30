from app.core.chunking import chunk_text
from app.core.embeddings import EmbeddingModel
from app.core.vector_store import VectorStore
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def ingest_documents(texts: list[str]):
    embedder = EmbeddingModel()
    chunks = []
    metadatas = []

    for i, text in enumerate(texts):
        text_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in text_chunks:
            chunks.append(chunk)
            metadatas.append({"source": f"doc_{i}", "text": chunk})

    vectors = embedder.embed_documents(chunks)
    store = VectorStore(vectors.shape[1])
    store.add(vectors, metadatas)

    return {
        "documents": len(texts),
        "chunks": len(chunks)
    }
