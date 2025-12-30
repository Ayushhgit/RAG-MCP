from pydantic import BaseModel
from typing import List, Dict, Any

class AnswerRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    text: str
    source: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    question: str
    agent_used: str = "general_qa"