import json
import datetime
import os
from typing import List, Dict, Any, Optional
from configs.config import settings
from utils.logger import logger

class RetrievalTracer:
    """
    Persistent logging for retrieval diagnostics and evaluation.
    Logs every query, its rewrite, retrieved hits, and scores.
    """
    def __init__(self, trace_file: str = "retrieval_traces.jsonl"):
        self.trace_path = settings.DATA_DIR / trace_file
        os.makedirs(settings.DATA_DIR, exist_ok=True)

    def log_trace(self, 
                  query: str, 
                  rewritten_query: Optional[str], 
                  hits: List[Dict[str, Any]], 
                  metadata: Optional[Dict[str, Any]] = None):
        """Logs a single retrieval event to a JSONL file."""
        trace_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "original_query": query,
            "rewritten_query": rewritten_query,
            "hits": [
                {
                    "chunk_id": h.get("chunk_id"),
                    "score": h.get("score"),
                    "dense_score": h.get("dense_score"),
                    "bm25_score": h.get("bm25_score"),
                    "rrf_score": h.get("rrf_score"),
                    "source": h.get("source")
                } for h in hits
            ],
            "metadata": metadata or {}
        }
        
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log retrieval trace: {e}")

tracer = RetrievalTracer()
