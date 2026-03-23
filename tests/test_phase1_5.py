import pytest
import asyncio
from pipeline.rag_pipeline import RAGPipeline
from generation.response_parser import RAGResponse

@pytest.mark.asyncio
async def test_end_to_end_rag():
    pipeline = RAGPipeline()
    
    # Test Ingestion
    text = "The capital of France is Paris. It is a major European city."
    source = "test_doc.txt"
    num_chunks = await pipeline.ingest_document(text, source)
    assert num_chunks > 0
    
    # Test Query
    query = "What is the capital of France?"
    response = await pipeline.query(query)
    
    assert isinstance(response, RAGResponse)
    assert "Paris" in response.answer
    assert response.confidence > 0.5
    assert len(response.sources) > 0

@pytest.mark.asyncio
async def test_low_relevance_query():
    pipeline = RAGPipeline()
    query = "Who won the World Cup in 1954?" # Not in context
    response = await pipeline.query(query)
    
    # Should ideally return "I don't have enough information" or similar based on system prompt
    assert "information" in response.answer.lower() or response.confidence < 0.5
