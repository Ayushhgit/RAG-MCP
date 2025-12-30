from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.core.llm import generate_answer, generate_answer_stream
from app.core.embeddings import EmbeddingModel
from app.core.router import QueryRouter, AgentType
from app.core.rag_planner import RAGPlanner
from app.core.tool_calling_agent import ToolCallingAgent
from app.core.context_compressor import ContextCompressor
from app.schemas.answer import AnswerRequest, AnswerResponse, SourceDocument
from app.utils.logger import logger
import asyncio

def answer_question(question: str, use_planner: bool = False, use_tool_calling: bool = False, stream: bool = False) -> AnswerResponse:
    """Answer a question using advanced RAG features."""
    try:
        logger.info(f"Answering question: {question}")

        # Initialize components
        embedder = EmbeddingModel()
        vector_dim = len(embedder.embed_query("test"))
        store = VectorStore(vector_dim)
        retriever = Retriever(store)
        context_compressor = ContextCompressor()

        # Use tool-calling agent if requested
        if use_tool_calling:
            agent = ToolCallingAgent()
            result = agent.execute_with_tools(question)
            return AnswerResponse(
                answer=result["answer"],
                sources=[],
                question=question,
                agent_used="tool_calling_agent"
            )

        # Use planner for complex queries if requested
        if use_planner:
            planner = RAGPlanner()
            result = planner.plan_and_execute(question, retriever)
            return AnswerResponse(
                answer=result["answer"],
                sources=[SourceDocument(text=s.get("text", ""), source=s.get("source", "")) for s in result.get("sources", [])],
                question=question,
                agent_used="multi_agent_planner"
            )

        # Standard RAG pipeline
        agent_type = QueryRouter().route_query(question)

        # Retrieve documents
        docs = retriever.retrieve(question)

        # Compress context if needed
        context = context_compressor.compress_context(question, docs)

        # Generate answer (streaming not supported in sync response)
        if stream:
            # For streaming, we'd need async handling - return note about streaming
            answer = generate_answer(context, question, agent_type) + " [Note: Streaming requested but not available in sync mode]"
        else:
            answer = generate_answer(context, question, agent_type)

        # Format sources
        sources = [
            SourceDocument(
                text=doc.get("text", ""),
                source=doc.get("source", "")
            )
            for doc in docs
        ]

        logger.info(f"Generated answer with {len(sources)} sources using {agent_type.value} agent")
        return AnswerResponse(
            answer=answer,
            sources=sources,
            question=question,
            agent_used=agent_type.value
        )
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise

async def answer_question_stream(question: str, use_planner: bool = False, use_tool_calling: bool = False) -> str:
    """Answer a question with streaming response."""
    try:
        logger.info(f"Streaming answer for question: {question}")

        # Initialize components
        embedder = EmbeddingModel()
        vector_dim = len(embedder.embed_query("test"))
        store = VectorStore(vector_dim)
        retriever = Retriever(store)
        context_compressor = ContextCompressor()

        # Use tool-calling agent if requested (streaming not supported)
        if use_tool_calling:
            agent = ToolCallingAgent()
            result = agent.execute_with_tools(question)
            yield result["answer"]
            return

        # Use planner for complex queries if requested
        if use_planner:
            planner = RAGPlanner()
            result = planner.plan_and_execute(question, retriever)
            yield result["answer"]
            return

        # Standard streaming RAG pipeline
        agent_type = QueryRouter().route_query(question)

        # Retrieve documents
        docs = retriever.retrieve(question)

        # Compress context if needed
        context = context_compressor.compress_context(question, docs)

        # Stream the answer
        async for chunk in generate_answer_stream(context, question, agent_type):
            yield chunk

    except Exception as e:
        logger.error(f"Error in streaming answer: {str(e)}")
        yield f"Error: {str(e)}"
