# RAG MCP Server

A Retrieval-Augmented Generation (RAG) system implemented as a Model Context Protocol (MCP) server with advanced features including hybrid search, reranking, multi-agent routing, context compression, streaming, and tool-calling agents.

## Features

- **Hybrid Search**: BM25 + Vector search + Reranking for superior retrieval
- **Reranking**: Cross-encoder based result reranking for improved relevance
- **Multi-agent Routing**: Intelligent query routing to specialized agents
- **Context Compression**: Automatic context compression to fit token limits
- **Streaming Answers**: Real-time streaming responses
- **Multi-agent RAG Planner**: Complex query decomposition and coordination
- **Tool-calling Agent**: Dynamic tool usage with function calling
- Document ingestion and chunking
- Vector embeddings using Sentence Transformers
- FAISS vector storage with persistence
- Groq LLM integration
- MCP server interface for tool integration

## Advanced Features

### Hybrid Search (BM25 + Vector + Rerank)
The system combines three retrieval methods for optimal results:
1. **BM25**: Lexical search using term frequency-inverse document frequency
2. **Vector Search**: Semantic search using dense embeddings
3. **Reranking**: Cross-encoder reranking for final relevance scoring

### Multi-Agent Router
Queries are automatically routed to specialized agents based on content analysis:
- **General QA**: Factual questions and explanations
- **Technical**: API, infrastructure, and technical topics
- **Creative**: Brainstorming, design, and creative tasks
- **Code**: Programming questions and code-related queries
- **Math**: Mathematical problems and calculations

### Multi-Agent RAG Planner
For complex queries, the planner:
- Decomposes queries into sub-queries
- Routes each sub-query to appropriate agents
- Synthesizes coherent final answers
- Handles comparative and multi-part questions

### Context Compression
Automatically compresses retrieved context using:
- **Extractive Compression**: Selects most relevant sentences
- **Abstractive Compression**: Uses LLM to summarize content
- Configurable compression ratios and length limits

### Tool-Calling Agent
An intelligent agent that can dynamically call RAG tools:
- Health checks
- Document ingestion
- Knowledge search
- Question answering
- Multi-turn conversations with tool usage

### Streaming Answers
Real-time streaming responses for better user experience with long-form content.
