import asyncio
import time
from streaming.producer import producer
from streaming.worker import AegisWorker
from pipeline.rag_pipeline import RAGPipeline
from evaluation.feedback_manager import feedback_manager
from utils.logger import logger

async def run_phase3_demo():
    """
    Demonstrates the full Phase 3 Elite flow:
    Streaming -> Incremental Indexing -> Temporal Decay -> Feedback Boost.
    """
    logger.info("🚀 Starting Phase 3 Elite Verification...")
    
    # 1. Initialize Pipeline
    pipeline = RAGPipeline()
    
    # 2. Produce Streaming Events (Simulating real-time data)
    logger.info("--- Step 1: Streaming Ingestion ---")
    producer.produce(
        doc_id="news_01", 
        content="AegisLLM released a new update for continual learning systems today.",
        metadata={"source": "tech_blog", "category": "news"}
    )
    
    await asyncio.sleep(2) # Give consumer time
    
    # 3. Querying (Should reflect new data immediately)
    logger.info("--- Step 2: Real-time Retrieval ---")
    result = await pipeline.query("What is the latest update for AegisLLM?")
    logger.info(f"Retrieval Response: {result['answer']}")
    
    # 4. Demonstrate Temporal Decay
    logger.info("--- Step 3: Temporal Prioritization ---")
    # Add an older identical doc (simulated by manual timestamp if we had it, 
    # but here we just show the score field)
    hits = result.get("retrieval_trace", {}).get("hits", [])
    if hits:
        logger.info(f"Top Hit Freshness Score: {hits[0].get('freshness_score')}")
    
    # 5. Demonstrate Feedback Loop
    logger.info("--- Step 4: Implicit Learning (Feedback) ---")
    if hits:
        feedback_manager.log_feedback(
            query="What is the latest update for AegisLLM?",
            doc_id="news_01",
            rating=5
        )
        
        # Re-query to observe boost
        result_v2 = await pipeline.query("What is the latest update for AegisLLM?")
        hits_v2 = result_v2.get("retrieval_trace", {}).get("hits", [])
        if hits_v2:
            logger.info(f"Doc 'news_01' Feedback Boost: {hits_v2[0].get('feedback_boost')}")

    logger.info("✅ Phase 3 Elite Verification Complete.")

if __name__ == "__main__":
    # Start worker in background (simulated for demo)
    # In reality, worker runs in a separate process
    asyncio.run(run_phase3_demo())
