import numpy as np
from typing import List, Dict, Any
from ingestion.embedder import EmbeddingClient
from utils.logger import logger, observe_latency
from configs.config import settings

class ContextCompressor:
    """
    Elite Context Optimization: Redundancy elimination and informative pruning.
    """
    def __init__(self, embedder: EmbeddingClient):
        self.embedder = embedder
        self.threshold = settings.CONTEXT_COMPRESSION_THRESHOLD

    @observe_latency
    def compress(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prunes redundant hits based on semantic similarity.
        Keep the highest scoring hit among a group of similar ones.
        """
        if not hits: return []

        # 1. Get embeddings for all hits
        contents = [h["content"] for h in hits]
        embeddings = self.embedder.get_embeddings(contents)
        
        # 2. Sequential filtering (Greedy redundancy removal)
        selected_hits = []
        selected_embeddings = []

        for i, (hit, emb) in enumerate(zip(hits, embeddings)):
            is_redundant = False
            
            for prev_emb in selected_embeddings:
                # Cosine similarity
                similarity = np.dot(emb, prev_emb)
                if similarity > self.threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_hits.append(hit)
                selected_embeddings.append(emb)
            else:
                logger.debug(f"Pruned redundant chunk: {hit['chunk_id']}")

        logger.info(f"Context compressed: {len(hits)} -> {len(selected_hits)} chunks.")
        return selected_hits

    def prune_sentences(self, text: str, query_emb: np.ndarray, top_n: int = 3) -> str:
        """
        Advanced: Keep only the most relevant sentences within a single chunk.
        """
        sentences = text.split(". ")
        if len(sentences) <= top_k: return text
        
        s_embs = self.embedder.get_embeddings(sentences)
        scores = np.dot(s_embs, query_emb)
        
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = sorted([x[0] for x in indexed_scores[:top_n]])
        
        return ". ".join([sentences[i] for i in top_indices])
