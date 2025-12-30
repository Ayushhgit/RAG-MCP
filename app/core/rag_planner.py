from typing import List, Dict, Any, Optional
from app.core.router import QueryRouter, AgentType
from app.core.llm import generate_answer
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.core.context_compressor import ContextCompressor
from app.utils.logger import logger
import re

class RAGPlanner:
    def __init__(self):
        self.router = QueryRouter()
        self.context_compressor = ContextCompressor()

    def plan_and_execute(self, query: str, retriever: Retriever) -> Dict[str, Any]:
        """Plan and execute a complex query using multiple agents."""
        try:
            # Analyze query complexity
            sub_queries = self._decompose_query(query)

            if len(sub_queries) <= 1:
                # Simple query - use single agent
                return self._execute_simple_query(query, retriever)

            # Complex query - coordinate multiple agents
            logger.info(f"Decomposed complex query into {len(sub_queries)} sub-queries")
            return self._execute_complex_query(query, sub_queries, retriever)

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._execute_simple_query(query, retriever)

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex queries into simpler sub-queries."""
        # Simple decomposition based on keywords and structure
        sub_queries = []

        # Check for comparative queries
        if any(word in query.lower() for word in ['compare', 'versus', 'vs', 'difference between']):
            parts = re.split(r'compare|versus|vs|difference between', query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                sub_queries.extend([part.strip() for part in parts if part.strip()])

        # Check for multi-part questions
        elif any(word in query.lower() for word in ['and', 'also', 'additionally']):
            parts = re.split(r'and|also|additionally', query, flags=re.IGNORECASE)
            sub_queries.extend([part.strip() + "?" for part in parts if part.strip()])

        # Check for step-by-step queries
        elif any(word in query.lower() for word in ['how to', 'steps to', 'guide for']):
            # Keep as single query but mark as procedural
            sub_queries = [query]

        else:
            # Default: treat as single query
            sub_queries = [query]

        return [q for q in sub_queries if q]

    def _execute_simple_query(self, query: str, retriever: Retriever) -> Dict[str, Any]:
        """Execute a simple query with single agent."""
        agent_type = self.router.route_query(query)

        # Retrieve documents
        docs = retriever.retrieve(query)

        # Compress context if needed
        context = self.context_compressor.compress_context(query, docs)

        # Generate answer
        answer = generate_answer(context, query, agent_type)

        return {
            "answer": answer,
            "agent_used": agent_type.value,
            "sources": docs,
            "query_type": "simple",
            "sub_queries": 1
        }

    def _execute_complex_query(self, original_query: str, sub_queries: List[str], retriever: Retriever) -> Dict[str, Any]:
        """Execute a complex query by coordinating multiple agents."""
        sub_answers = []
        all_sources = []

        for i, sub_query in enumerate(sub_queries):
            logger.info(f"Processing sub-query {i+1}/{len(sub_queries)}: {sub_query}")

            # Route each sub-query to appropriate agent
            agent_type = self.router.route_query(sub_query)

            # Retrieve documents for this sub-query
            docs = retriever.retrieve(sub_query)
            all_sources.extend(docs)

            # Compress context
            context = self.context_compressor.compress_context(sub_query, docs)

            # Generate answer for sub-query
            sub_answer = generate_answer(context, sub_query, agent_type)

            sub_answers.append({
                "sub_query": sub_query,
                "answer": sub_answer,
                "agent": agent_type.value,
                "sources": docs
            })

        # Synthesize final answer
        final_answer = self._synthesize_answers(original_query, sub_answers)

        return {
            "answer": final_answer,
            "agent_used": "multi_agent_planner",
            "sources": all_sources,
            "query_type": "complex",
            "sub_queries": len(sub_queries),
            "sub_answers": sub_answers
        }

    def _synthesize_answers(self, original_query: str, sub_answers: List[Dict[str, Any]]) -> str:
        """Synthesize a coherent answer from multiple sub-answers."""
        try:
            # Create synthesis prompt
            synthesis_prompt = f"""
            Original Query: {original_query}

            I have gathered information from multiple specialized agents. Please synthesize a coherent, comprehensive answer:

            {"".join([f"Sub-query: {sa['sub_query']}\nAnswer: {sa['answer']}\n\n" for sa in sub_answers])}

            Provide a unified answer that addresses the original query comprehensively:
            """

            # Use general QA agent for synthesis
            synthesized = generate_answer("", synthesis_prompt, AgentType.GENERAL_QA)
            return synthesized

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback: concatenate answers
            return "\n\n".join([f"Regarding '{sa['sub_query']}': {sa['answer']}" for sa in sub_answers])