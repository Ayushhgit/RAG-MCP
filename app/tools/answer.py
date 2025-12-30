from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.core.llm import generate_answer
from app.core.embeddings import EmbeddingModel
from app.schemas.answer import AnswerRequest, AnswerResponse, SourceDocument
from app.utils.logger import logger

def answer_question(question: str) -> AnswerResponse:
    """Answer a question using the RAG system."""
    try:
        logger.info(f"Answering question: {question}")

        # Initialize components
        embedder = EmbeddingModel()
        vector_dim = len(embedder.embed_query("test"))
        store = VectorStore(vector_dim)
        retriever = Retriever(store)

        # Retrieve relevant documents
        raw_docs = retriever.retrieve(question)

        # Format context
        context = "\n".join([d.get("text", "") for d in raw_docs])

        # Generate answer
        answer = generate_answer(context, question)

        # Format sources
        sources = [
            SourceDocument(
                text=doc.get("text", ""),
                source=doc.get("source", "")
            )
            for doc in raw_docs
        ]

        logger.info(f"Generated answer with {len(sources)} sources")
        return AnswerResponse(
            answer=answer,
            sources=sources,
            question=question
        )
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise
