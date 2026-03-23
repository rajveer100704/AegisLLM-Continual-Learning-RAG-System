from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the core features of AegisLLM?")

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source: str = Field(..., example="technical_docs_v1")

class FeedbackRequest(BaseModel):
    query: str
    doc_id: str
    rating: int = Field(..., ge=1, le=5) # 1-5 scale

class MetricsResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    total_queries: int
    safety_violations: int
    avg_latency_ms: float
