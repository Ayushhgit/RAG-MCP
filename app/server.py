import asyncio
from mcp import Tool
from mcp.server import Server
from mcp.types import TextContent, PromptMessage
import mcp.server.stdio
from app.tools.health import health_check
from app.tools.ingest import ingest_documents
from app.tools.search import search_knowledge
from app.tools.answer import answer_question
from app.schemas.ingest import IngestRequest
from app.schemas.search import SearchRequest
from app.schemas.answer import AnswerRequest

server = Server("rag-mcp")

@server.tool()
async def health() -> list[TextContent]:
    """Check the health status of the RAG system."""
    result = health_check()
    return [TextContent(type="text", text=str(result))]

@server.tool()
async def ingest(documents: str) -> list[TextContent]:
    """Ingest documents into the vector store. Documents should be a JSON array of strings."""
    import json
    try:
        docs = json.loads(documents)
        result = ingest_documents(docs)
        return [TextContent(type="text", text=str(result))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

@server.tool()
async def search(query: str, top_k: int = 5) -> list[TextContent]:
    """Search the knowledge base for relevant documents."""
    try:
        results = search_knowledge(query)
        return [TextContent(type="text", text=str(results))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

@server.tool()
async def answer(question: str, use_planner: bool = False, use_tool_calling: bool = False, stream: bool = False) -> list[TextContent]:
    """Answer a question using advanced RAG features.
    - use_planner: Use multi-agent planner for complex queries
    - use_tool_calling: Use tool-calling agent for dynamic tool usage
    - stream: Enable streaming response (async)
    """
    try:
        if stream:
            # For streaming, we'd need async handling - not supported in current MCP setup
            result = answer_question(question, use_planner, use_tool_calling, stream=False)
            return [TextContent(type="text", text=f"Streaming not supported in MCP. Answer: {result.answer}")]
        else:
            result = answer_question(question, use_planner, use_tool_calling, stream)
            return [TextContent(type="text", text=str(result))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())