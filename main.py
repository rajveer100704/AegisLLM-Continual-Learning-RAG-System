import asyncio
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import logger
import os

async def main():
    # Initialize Pipeline
    pipeline = RAGPipeline()
    
    # Sample Document for Ingestion
    sample_text = """
    AegisLLM is a state-of-the-art Continual Learning RAG system designed for production.
    It uses hybrid retrieval (dense + sparse) and streaming adaptation via Redis Streams.
    The system is built with FastAPI and uses Gemini 2.5 Pro for generation.
    Key features include context compression, token budgeting, and a dedicated safety layer.
    The system architecture follows a modular design for independent scalability.
    """
    
    source = "sample_manual.txt"
    
    # 1. Ingestion
    logger.info("--- Starting Ingestion ---")
    await pipeline.ingest_document(sample_text, source)
    
    # 2. Query
    query = "What is AegisLLM and what are its key features?"
    logger.info(f"--- Processing Query: {query} ---")
    
    try:
        response = await pipeline.query(query)
        
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("-" * 50)
        print(f"ANSWER: {response.answer}")
        print(f"SOURCES: {', '.join(response.sources)}")
        print(f"CONFIDENCE: {response.confidence:.2f}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
