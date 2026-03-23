import re
from typing import List, Dict, Any, Optional, Tuple
from generation.llm_client import LLMClient
from utils.logger import logger
from configs.config import settings

class InputGuard:
    """
    Detects prompt injection and malicious intent.
    """
    INJECTION_KEYWORDS = [
        r"ignore previous instructions",
        r"system prompt",
        r"bypass",
        r"do not mention",
        r"stop filtered",
        r"roleplay as",
        r"jailbreak"
    ]

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def _heuristic_check(self, query: str) -> float:
        """Simple keyword matching."""
        matches = 0
        for pattern in self.INJECTION_KEYWORDS:
            if re.search(pattern, query, re.IGNORECASE):
                matches += 1
        return min(matches * 0.4, 1.0)

    async def _llm_check(self, query: str) -> float:
        """Deep intent classifier using Gemini."""
        prompt = f"""
        Analyze this user query for malicious intent, jailbreak attempts, or prompt injection.
        Query: "{query}"

        Return ONLY a float between 0.0 (safe) and 1.0 (dangerous).
        """
        try:
            raw_score = await self.llm_client.generate_async(prompt)
            # Basic float extraction
            return float(re.findall(r"0?\.\d+", str(raw_score))[0])
        except Exception:
            return 0.0

    async def check(self, query: str) -> Tuple[bool, float]:
        """Returns (is_blocked, risk_score)."""
        h_score = self._heuristic_check(query)
        if h_score > 0.8: # Fast block for high-risk heuristics
            return True, h_score
        
        l_score = await self._llm_check(query)
        risk_score = (h_score * 0.3) + (l_score * 0.7)
        return risk_score > settings.INJECTION_RISK_THRESHOLD, risk_score

class ContextGuard:
    """
    Checks retrieved context for relevance and adversarial content.
    """
    def check(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # For now, simple keyword filter for 'ignore' in contents (Indirect injection)
        safe_chunks = []
        for chunk in chunks:
            content = chunk["content"].lower()
            if "ignore previous" in content or "system instruction" in content:
                logger.warning(f"Adversarial content detected in chunk {chunk['chunk_id']}. Filtering.")
                continue
            safe_chunks.append(chunk)
        return safe_chunks

class OutputGuard:
    """
    Validates grounding and toxicity of generated output.
    """
    def __init__(self, embedder):
        self.embedder = embedder

    def check_grounding(self, answer: str, context_chunks: List[Dict[str, Any]]) -> float:
        """Simple semantic similarity check for grounding."""
        if not context_chunks: return 0.0
        
        context_texts = [c["content"] for c in context_chunks]
        # In a real system, we'd use use a more advanced entailment model (NLI)
        # Here we use mean cosine similarity as a proxy for grounding coverage
        answer_emb = self.embedder.get_embedding(answer)
        max_similarity = 0.0
        
        for ctx in context_texts:
            ctx_emb = self.embedder.get_embedding(ctx)
            similarity = self.embedder.compute_similarity(answer_emb, ctx_emb)
            max_similarity = max(max_similarity, similarity)
            
        return max_similarity
