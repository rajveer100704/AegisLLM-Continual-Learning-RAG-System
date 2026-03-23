import asyncio
import os
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import logger

async def manual_bench():
    pipeline = RAGPipeline()
    
    # 1. Ingest test data (Minimal)
    texts = [
        "AegisLLM uses RRF for hybrid fusion of dense and sparse signals.",
        "The reciprocal rank fusion (RRF) formula is 1 / (k + rank)."
    ]
    for i, txt in enumerate(texts):
        await pipeline.ingest_document(txt, f"bench_doc_{i}.txt")
    
    # Delay to avoid rate limit
    await asyncio.sleep(5)
    
    # 2. Test Query Expansion & Guardrails (One focused query)
    query = "rrf formula"
    logger.info(f"--- Testing Query: {query} ---")
    response = await pipeline.query(query)
    
    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print(f"ANSWER: {response.answer}")
    print(f"SOURCES: {response.sources}")
    print(f"CONFIDENCE: {response.confidence:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(manual_bench())
