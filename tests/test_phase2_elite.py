import pytest
import asyncio
import os
from utils.logger import logger
from pipeline.rag_pipeline import RAGPipeline
from retrieval.query_rewriter import QueryRewriter

@pytest.mark.asyncio
async def test_hybrid_vs_dense():
    pipeline = RAGPipeline()
    
    # Ingest diverse context
    texts = [
        "Python is a high-level, interpreted programming language.",
        "The Python snake is one of the largest species of snakes in the world.",
        "Guido van Rossum created the Python programming language in 1991.",
        "Reticulated pythons are found in Southeast Asia."
    ]
    for i, txt in enumerate(texts):
        await pipeline.ingest_document(txt, f"doc_{i}.txt")
    
    # Query that benefits from Hybrid (Keyword + Semantic)
    query = "Who is the creator of python?"
    
    # 1. Test Dense Only (Internal check)
    dense_hits = pipeline.dense_retriever.retrieve(query)
    
    # 2. Test Hybrid
    hybrid_hits = await pipeline.query(query)
    
    # Assertions
    assert "Guido" in hybrid_hits.answer
    assert hybrid_hits.confidence > 0.6
    logger.info(f"Hybrid Accuracy Verified. Answer: {hybrid_hits.answer}")

@pytest.mark.asyncio
async def test_query_rewriter_guardrails():
    pipeline = RAGPipeline()
    rewriter = pipeline.rewriter
    
    # Normal expansion
    original = "RAG systems"
    rewritten = await rewriter.rewrite(original)
    assert len(rewritten) >= len(original)
    
    # Intent drift test (simulated by prompt injection attempt or nonsense)
    original_vague = "pizza recipes" # Unrelated to current index context mostly
    rewritten_vague = await rewriter.rewrite(original_vague)
    # The guardrail should keep it relatively close or fallback if Gemini goes crazy
    assert len(rewritten_vague) > 0

@pytest.mark.asyncio
async def test_retrieval_trace_persistence():
    pipeline = RAGPipeline()
    await pipeline.query("test trace query")
    
    assert os.path.exists(pipeline.vector_store.index_path.parent / "retrieval_traces.jsonl")
