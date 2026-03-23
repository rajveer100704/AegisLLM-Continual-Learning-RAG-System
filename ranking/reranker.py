from typing import List, Dict, Any
import numpy as np
from ingestion.embedder import EmbeddingClient
from utils.logger import logger, observe_latency

class Reranker:
    """
    Bi-encoder reranking layer for secondary scoring and calibration.
    """
    def __init__(self, embedder: EmbeddingClient):
        self.embedder = embedder

    @observe_latency
    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank hits with bi-encoder similarity and implicit feedback signals.
        """
        if not hits: return []
        
        # 1. Base Reranking (Bi-encoder similarity)
        query_emb = self.embedder.get_embeddings(query)[0]
        
        # 2. Integrate Feedback Signals
        from evaluation.feedback_manager import feedback_manager
        boosts = feedback_manager.get_boost_scores()
        import math

        for hit in hits:
            # Re-calculate similarity for consistency
            hit_emb = self.embedder.get_embeddings(hit["content"])[0]
            similarity = float(np.dot(query_emb, hit_emb))
            
            # Application of "Implicit Learning" boost
            did = hit.get("doc_id", "")
            boost_val = boosts.get(did, 0.0)
            multiplier = 1.0 + (0.2 * math.log1p(boost_val))
            
            hit["rerank_score"] = similarity * multiplier
            hit["feedback_boost"] = multiplier
            # Update primary score for generator
            hit["score"] = hit["rerank_score"]
            
        # 3. Sort by final rerank score
        sorted_hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Reranking complete for {len(hits)} hits with feedback signals.")
        return sorted_hits
