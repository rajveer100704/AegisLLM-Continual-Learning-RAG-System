import json
import os
from typing import List, Dict, Any
from utils.logger import logger
from configs.config import settings

class FeedbackManager:
    """
    Manages user feedback logs and provides signals for reranking optimization.
    """
    def __init__(self):
        self.feedback_path = settings.DATA_DIR / "feedback_logs.jsonl"
        self._ensure_file()

    def _ensure_file(self):
        if not self.feedback_path.exists():
            self.feedback_path.touch()

    def log_feedback(self, query: str, doc_id: str, rating: int, metadata: Dict[str, Any] = None):
        """Log user feedback (e.g. click/rating)."""
        entry = {
            "query": query,
            "doc_id": doc_id,
            "rating": rating, # 1 to 5
            "timestamp": os.path.getmtime(self.feedback_path), # Simple timestamp
            "metadata": metadata or {}
        }
        with open(self.feedback_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Feedback logged for doc {doc_id} on query '{query[:20]}...'")

    def get_boost_scores(self) -> Dict[str, float]:
        """
        Analyze logs and return a boost map for high-rated documents.
        Simple logic: avg_rating * frequency.
        """
        boosts = {}
        if not self.feedback_path.exists(): return boosts
        
        try:
            with open(self.feedback_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    did = entry["doc_id"]
                    score = entry["rating"] / 5.0 # Normalize to 0-1
                    boosts[did] = boosts.get(did, 0.0) + score
        except Exception as e:
            logger.error(f"Error reading feedback logs: {e}")
            
        return boosts

feedback_manager = FeedbackManager()
