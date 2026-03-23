from typing import List, Dict, Any, Optional
from ingestion.chunker import TextChunker
from ingestion.embedder import EmbeddingClient
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_rewriter import QueryRewriter
from ranking.reranker import Reranker
from generation.llm_client import LLMClient
from generation.generator import Generator, RAGResponse
from generation.guardrails import InputGuard, ContextGuard
from evaluation.retrieval_trace import tracer
from utils.logger import logger, observe_latency_async
from configs.config import settings

class RAGPipeline:
    """
    Elite Pipeline Orchestrator with 3-Layer Safety Shield.
    """
    def __init__(self, vector_store: Optional[VectorStore] = None):
        # 1. Base Components
        self.chunker = TextChunker()
        self.embedder = EmbeddingClient()
        self.vector_store = vector_store or VectorStore()
        
        # 2. Retrieval Suite
        self.dense_retriever = Retriever(self.vector_store, self.embedder)
        self.sparse_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(self.dense_retriever, self.sparse_retriever)
        
        # 3. Intelligence & Safety Layers
        self.llm_client = LLMClient()
        self.rewriter = QueryRewriter(self.llm_client, self.embedder)
        self.reranker = Reranker(self.embedder)
        self.input_guard = InputGuard(self.llm_client)
        self.context_guard = ContextGuard()
        
        # 4. Generation
        self.generator = Generator(self.llm_client, embedder=self.embedder)
        
        # Hydrate BM25 if index exists to keep parity
        if self.vector_store.metadata_map:
            self.sparse_retriever.index(list(self.vector_store.metadata_map.values()))
            
        logger.info("AegisLLM Elite Pipeline Orchestrator initialized.")

    @observe_latency_async
    async def ingest_document(self, text: str, source: str) -> int:
        """Full ingestion flow: Chunk -> Embed -> Index -> BM25."""
        logger.info(f"Ingesting document from source: {source}")
        
        # 1. Chunking
        chunks = self.chunker.chunk_text(text, source=source)
        
        # 2. Embedding
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.get_embeddings(texts)
        
        # 3. Dense Indexing
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save()
        
        # 4. Sparse Indexing (Incremental update for BM25 - in memory for now)
        self.sparse_retriever.index(list(self.vector_store.metadata_map.values()))
        
        logger.info(f"Ingestion complete. Total chunks: {len(chunks)}")
        return len(chunks)

    @observe_latency_async
    async def query(self, query_text: str) -> RAGResponse:
        """Full Elite RAG flow with 3-Layer Safety (Shield)."""
        logger.info(f"Processing Elite query: {query_text}")
        
        # 1. Input Guard (Phase 6)
        if settings.ENABLE_SAFETY_GUARDRAILS:
            is_blocked, risk_score = await self.input_guard.check(query_text)
            if is_blocked:
                logger.warning(f"Query blocked by InputGuard (Risk: {risk_score:.2f})")
                from evaluation.safety_monitor import safety_monitor
                safety_monitor.log_violation("input_injection", query_text, risk_score)
                return RAGResponse(answer=settings.SAFETY_BLOCK_RESPONSE, sources=[], confidence=0.0)

        # 2. Query Rewriting (Guardrailed)
        rewritten_query = await self.rewriter.rewrite(query_text)
        
        # 3. Hybrid Retrieval (RRF + Explainability)
        hybrid_hits = self.hybrid_retriever.retrieve(rewritten_query)
        
        # 4. Context Guard (Phase 6)
        if settings.ENABLE_SAFETY_GUARDRAILS:
            hybrid_hits = self.context_guard.check(rewritten_query, hybrid_hits)
            if not hybrid_hits:
                return RAGResponse(answer="No safe context found for this query.", sources=[], confidence=0.0)

        # 5. Reranking (Bi-encoder)
        final_hits = self.reranker.rerank(rewritten_query, hybrid_hits)
        
        # 6. Trace Logging
        tracer.log_trace(query_text, rewritten_query, final_hits)
        
        # 7. Generation (Inclusive of Output Guard)
        try:
            response = await self.generator.generate_response(rewritten_query, final_hits)
            return response
        except Exception as e:
            logger.error(f"Pipeline Generation Failure: {e}")
            return RAGResponse(
                answer="I'm sorry, I'm having trouble synthesizing an answer right now. Please try again or rephrase.",
                sources=[],
                confidence=0.0
            )
