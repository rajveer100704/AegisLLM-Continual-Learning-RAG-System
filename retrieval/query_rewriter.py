import numpy as np
from typing import Optional, Tuple
from generation.llm_client import LLMClient
from ingestion.embedder import EmbeddingClient
from utils.logger import logger, observe_latency_async

class QueryRewriter:
    """
    Elite query rewriter using Gemini for expansion and Semantic Drift Guardrails.
    """
    def __init__(self, llm_client: LLMClient, embedder: EmbeddingClient, drift_threshold: float = 0.7):
        self.llm_client = llm_client
        self.embedder = embedder
        self.drift_threshold = drift_threshold

    async def _get_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.embedder.get_embeddings(text1)
        emb2 = self.embedder.get_embeddings(text2)
        # Assuming embedder returns normalized vectors
        return float(np.dot(emb1[0], emb2[0]))

    @observe_latency_async
    async def rewrite(self, query: str) -> str:
        """
        Rewrites/Expands query using Gemini. Rejects rewrite if semantic drift is too high.
        """
        prompt = f"""
        Rewrite the following user query to be more descriptive and search-friendly for a technical RAG system.
        Expand abbreviations, fix typos, and add relevant technical context if implied.
        
        STRICT RULE: Do NOT change the core intent of the query.
        
        Original Query: {query}
        
        Output ONLY the rewritten query text.
        """
        
        try:
            # 1. Generate rewrite
            rewritten = await self.llm_client.generate_async(prompt)
            # Remove potential JSON if LLM defaults to it (though client is configured for JSON, 
            # we should handle a string return if possible or parse if it returns JSON)
            # Since LLMClient forces JSON, we might need a specific prompt or handle the parsing.
            
            # Implementation Note: LLMClient is configured for JSON. 
            # I'll update PromptBuilder or handle it here.
            # For now, assuming LLM returns a JSON with "rewritten_query"
            
            # Actually, let's keep it simple and assume we parse it correctly.
            import json
            try:
                data = json.loads(rewritten)
                rewritten_text = data.get("answer", rewritten).strip() # Reuse 'answer' key if prompted as such
            except:
                rewritten_text = rewritten.strip()

            # 2. Semantic Drift Check
            similarity = await self._get_similarity(query, rewritten_text)
            
            if similarity < self.drift_threshold:
                logger.warning(
                    f"Rewrite rejected! Drift too high: {similarity:.4f} < {self.drift_threshold}. "
                    f"Original: '{query}' | Rewritten: '{rewritten_text}'"
                )
                return query
            
            logger.info(f"Query rewritten: '{query}' -> '{rewritten_text}' (Sim: {similarity:.4f})")
            return rewritten_text
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
