from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any
from utils.logger import logger, observe_latency

class BM25Retriever:
    """
    Sparse retriever using BM25 algorithm for keyword-based search.
    """
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.corpus_tokenized = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenization."""
        return re.findall(r'\w+', text.lower())

    @observe_latency
    def index(self, chunks: List[Dict[str, Any]]):
        """Indexes a list of chunks for BM25 search."""
        if not chunks: return
        self.chunks = chunks
        self.corpus_tokenized = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokenized)
        logger.info(f"Indexed {len(chunks)} chunks for BM25.")

    @observe_latency
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks using BM25 score."""
        if not self.bm25: return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_n = min(len(scores), top_k)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        results = []
        for i in top_indices:
            res = self.chunks[i].copy()
            res["bm25_score"] = float(scores[i])
            results.append(res)
            
        return results
