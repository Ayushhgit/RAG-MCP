from pydantic import BaseModel
from typing import List, Dict, Any

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    source: str
    score: float = 0.0

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str