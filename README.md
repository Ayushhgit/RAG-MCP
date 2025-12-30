# RAG MCP Server

A Retrieval-Augmented Generation (RAG) system implemented as a Model Context Protocol (MCP) server.

## Features

- Document ingestion and chunking
- Vector embeddings using Sentence Transformers
- FAISS vector storage
- Groq LLM integration for answer generation
- MCP server interface for tool integration

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Create a `.env` file in the `app/` directory with your configuration:
   ```
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=llama3-8b-8192
   CHUNK_SIZE=512
   CHUNK_OVERLAP=50
   TOP_K=5
   ```

## Usage

Run the MCP server:
```bash
python main.py
```

## MCP Tools

The server provides the following tools:

- `health`: Check system health and statistics
- `ingest`: Ingest documents into the vector store
- `search`: Search for relevant documents
- `answer`: Answer questions using RAG

## Project Structure

```
rag-mcp/
├── main.py                 # Entry point
├── pyproject.toml         # Project configuration
├── README.md              # This file
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── server.py          # MCP server implementation
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chunking.py    # Text chunking utilities
│   │   ├── embeddings.py  # Embedding model wrapper
│   │   ├── llm.py         # LLM integration
│   │   ├── retriever.py   # Document retrieval
│   │   └── vector_store.py # FAISS vector store
│   ├── resources/
│   │   ├── __init__.py
│   │   └── stats.py       # System statistics
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── answer.py      # Answer API schemas
│   │   ├── ingest.py      # Ingest API schemas
│   │   └── search.py      # Search API schemas
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── answer.py      # Answer tool
│   │   ├── health.py      # Health check tool
│   │   ├── ingest.py      # Document ingestion tool
│   │   └── search.py      # Search tool
│   └── utils/
│       ├── __init__.py
│       ├── logger.py      # Logging utilities
│       └── ytils.py       # General utilities
└── data/
    ├── index/             # Vector index and metadata
    ├── processed/         # Processed data
    └── raw/               # Raw input data
```

## Development

The system is built with:
- Python 3.11+
- FAISS for vector search
- Sentence Transformers for embeddings
- Groq for LLM
- MCP for tool interface
- Pydantic for data validation