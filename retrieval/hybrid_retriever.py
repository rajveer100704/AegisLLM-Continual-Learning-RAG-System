from typing import List, Dict, Any
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from configs.config import settings
from utils.logger import logger, observe_latency
from evaluation.retrieval_trace import tracer

class HybridRetriever:
    """
    Elite Hybrid Retriever using Reciprocal Rank Fusion (RRF) with explainability.
    Combines Dense (FAISS) and Sparse (BM25) results.
    """
    def __init__(self, dense_retriever: Retriever, sparse_retriever: BM25Retriever, rrf_k: int = 60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
        self.top_k = settings.TOP_K

    def _calculate_temporal_score(self, ingestion_time_iso: str) -> float:
        """Calculate exponential decay freshness score [0, 1]."""
        import datetime
        import math
        try:
            ingestion_time = datetime.datetime.fromisoformat(ingestion_time_iso)
            now = datetime.datetime.utcnow()
            delta_hours = (now - ingestion_time).total_seconds() / 3600
            # score = exp(-lambda * delta_t)
            return math.exp(-settings.TEMPORAL_DECAY_RATE * delta_hours)
        except:
            return 0.5 # Default for missing/invalid time

    @observe_latency
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Fuse Dense + Sparse results with Temporal Freshness.
        final_score = alpha * rrf_score + (1-alpha) * freshness_score
        """
        top_k = top_k or self.top_k
        alpha = settings.HYBRID_FRESHNESS_WEIGHT
        
        # 1. Independent Retrieval
        dense_hits = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        sparse_hits = self.sparse_retriever.search(query, top_k=top_k * 2)
        
        # 2. RRF Fusion
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            chunk_map[cid] = hit

        for rank, hit in enumerate(sparse_hits):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = hit

        # 3. Apply Temporal Weighting
        final_results = []
        for cid, rrf_score in rrf_scores.items():
            hit = chunk_map[cid]
            
            # Normalize RRF score (approximate to 0-1 for hybrid sum)
            # Max possible RRF for 2 lists is (1/k + 1/k) = 2/60 = 0.033
            norm_rrf = rrf_score / (2.0 / self.rrf_k)
            
            # Freshness score
            freshness = self._calculate_temporal_score(hit.get("ingestion_timestamp", ""))
            
            # Elite Hybrid Formula
            final_h_score = (alpha * norm_rrf) + ((1 - alpha) * freshness)
            
            hit["rrf_score"] = rrf_score
            hit["freshness_score"] = freshness
            hit["score"] = final_h_score
            final_results.append(hit)

        # 4. Sort and Finalize
        sorted_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        logger.info(f"Retrieved {len(sorted_results)} hybrid-temporal hits.")
        return sorted_results
