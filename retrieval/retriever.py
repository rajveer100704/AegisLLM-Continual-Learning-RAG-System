import numpy as np
from typing import List, Dict, Any
from retrieval.vector_store import VectorStore
from ingestion.embedder import EmbeddingClient
from configs.config import settings
from utils.logger import logger, observe_latency

class Retriever:
    """
    Production-grade retriever with score normalization and diagnostics.
    """
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingClient):
        self.vector_store = vector_store
        self.embedder = embedder
        self.score_threshold = settings.RETRIEVAL_SCORE_THRESHOLD

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Min-max normalize scores within the current result set to [0, 1]."""
        if not results: return []
        
        scores = [r["score"] for r in results]
        min_s, max_s = min(scores), max(scores)
        
        if max_s == min_s:
            for r in results: r["normalized_score"] = 1.0
            return results
            
        for r in results:
            r["normalized_score"] = (r["score"] - min_s) / (max_s - min_s)
            
        return results

    @observe_latency
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Embed query, search vector store, and normalize scores.
        """
        top_k = top_k or settings.TOP_K
        
        # 1. Generate query embedding
        query_embedding = self.embedder.get_embeddings(query)
        
        # 2. Search vector store
        raw_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 3. Score Normalization & Filtering
        # For IndexFlatIP, results are already cosine similarities if vectors were normalized.
        # We still apply min-max for relative ranking clarity.
        normalized_results = self._normalize_scores(raw_results)
        
        processed_results = []
        for res in normalized_results:
            logger.debug(f"Raw Score: {res['score']:.4f} | Norm Score: {res.get('normalized_score', 0):.4f} | ID: {res['chunk_id']}")
            
            # Use raw score for hard thresholding, but normalized for ranking display if needed
            if res["score"] >= self.score_threshold:
                processed_results.append(res)
            else:
                logger.warning(f"Filtered out chunk {res['chunk_id']} with low score: {res['score']:.4f}")

        # 4. Final diagnostics
        if not processed_results:
            logger.warning(f"No results found above threshold {self.score_threshold} for query: {query}")
        else:
            avg_score = sum(r["score"] for r in processed_results) / len(processed_results)
            logger.info(f"Retrieved {len(processed_results)} chunks (Filtered from {len(raw_results)}). Avg score: {avg_score:.4f}")

        return processed_results
