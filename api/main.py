import time
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from api.schemas import QueryRequest, IngestRequest, FeedbackRequest, MetricsResponse
from pipeline.rag_pipeline import RAGPipeline
from streaming.producer import StreamProducer
from evaluation.safety_monitor import safety_monitor
from utils.logger import logger
import asyncio

app = FastAPI(title="AegisLLM Athenaeum API", version="1.0.0")

# Singleton Pipeline
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

# Metrics tracking
START_TIME = time.time()
metrics = {"queries": 0, "violations": 0, "latencies": []}

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing AegisLLM Core...")
    get_pipeline()

@app.post("/athenaeum/query")
async def query_endpoint(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """Full Shield + Hybrid Retrieval + Synthesis."""
    start = time.time()
    try:
        metrics["queries"] += 1
        response = await pipeline.query(request.query)
        
        # Track latency
        duration = (time.time() - start) * 1000
        metrics["latencies"].append(duration)
        
        # Check if blocked
        if response.confidence == 0.0 and "blocked" in response.answer.lower():
            metrics["violations"] += 1

        return response
    except Exception as e:
        logger.error(f"API Query Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/athenaeum/ingest")
async def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """Triggers the streaming ingestion producer."""
    try:
        producer = StreamProducer()
        producer.push_document(request.text, {"source": request.source})
        return {"status": "accepted", "message": "Document queued for streaming ingestion."}
    except Exception as e:
        logger.error(f"Ingest Error: {e}")
        raise HTTPException(status_code=500, detail="Streaming producer failed.")

@app.post("/athenaeum/feedback")
async def feedback_endpoint(request: FeedbackRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """Informs the implicit learning loop."""
    try:
        # Route to Reranker learning logic
        pipeline.reranker.feedback_manager.log_feedback(
            query=request.query,
            doc_id=request.doc_id,
            signal="positive" if request.rating >= 4 else "negative"
        )
        return {"status": "success", "message": "Feedback captured. Reranker weights updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/athenaeum/metrics", response_model=MetricsResponse)
async def metrics_endpoint():
    avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"]) if metrics["latencies"] else 0
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": time.time() - START_TIME,
        "total_queries": metrics["queries"],
        "safety_violations": metrics["violations"],
        "avg_latency_ms": avg_latency
    }
