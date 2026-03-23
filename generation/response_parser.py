import json
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from utils.logger import logger

class RAGResponse(BaseModel):
    """Schema for validated RAG output."""
    answer: str
    sources: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

    @validator("answer")
    def answer_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v

class ResponseParser:
    """Parses and validates LLM raw JSON output."""
    
    @staticmethod
    def parse(raw_text: str) -> RAGResponse:
        """Parses raw JSON string into a validated RAGResponse model."""
        try:
            # Handle potential markdown code blocks in LLM output
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:-3].strip()
                
            data = json.loads(cleaned_text)
            return RAGResponse(**data)
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e} | Raw: {raw_text[:100]}...")
            # Fallback or re-raise
            raise
