from pydantic import BaseModel
from typing import List

class IngestRequest(BaseModel):
    documents: List[str]

class IngestResponse(BaseModel):
    documents: int
    chunks: int
    status: str = "success"