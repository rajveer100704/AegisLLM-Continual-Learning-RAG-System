from typing import List, Dict, Any
from generation.llm_client import LLMClient
from generation.prompt_builder import PromptBuilder
from generation.response_parser import ResponseParser, RAGResponse
from generation.context_compressor import ContextCompressor
from generation.summarizer import MapReduceSummarizer
from generation.guardrails import OutputGuard
from utils.logger import logger, observe_latency_async
from configs.config import settings

class Generator:
    """
    Orchestrates the LLM generation process with Elite Context Optimization and Safety.
    """
    def __init__(self, llm_client: LLMClient, embedder=None):
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        # Initialize optimizers & Safety
        self.compressor = ContextCompressor(embedder) if embedder else None
        self.summarizer = MapReduceSummarizer(llm_client)
        self.output_guard = OutputGuard(embedder) if embedder else None

    def _pack_context(self, chunks: List[Dict[str, Any]], max_tokens: int = None) -> List[Dict[str, Any]]:
        """Sort and pack chunks within budget."""
        max_tokens = max_tokens or settings.MAX_CONTEXT_TOKENS
        sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        packed_chunks = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            chunk_tokens = len(chunk["content"]) // 4
            if current_tokens + chunk_tokens <= max_tokens:
                packed_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        return packed_chunks

    @observe_latency_async
    async def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> RAGResponse:
        """Generates grounded response with hierarchical context optimization."""
        if not context_chunks:
            return RAGResponse(answer="I don't have enough information.", sources=[], confidence=0.0)

        # 1. Semantic Compression (Phase 4)
        processed_chunks = context_chunks
        if self.compressor:
            processed_chunks = self.compressor.compress(query, context_chunks)

        # 2. Adaptive Budgeting (Hierarchical Summarization)
        # If still too large, trigger Map-Reduce
        total_tokens = sum(len(c["content"]) for c in processed_chunks) // 4
        
        if total_tokens > settings.MAX_CONTEXT_TOKENS and settings.ENABLE_MAP_REDUCE:
            logger.info(f"Triggering Map-Reduce Summarization for large context ({total_tokens} tokens)")
            contents = [c["content"] for c in processed_chunks]
            summary = await self.summarizer.summarize_recursive(query, contents)
            # Create a virtual chunk for the summary
            final_context_chunks = [{"chunk_id": "summary_node", "content": summary, "score": 1.0}]
        else:
            final_context_chunks = self._pack_context(processed_chunks)

        # 3. Build & Call LLM
        # Use CoT for improved reasoning in Phase 4
        prompt = self.prompt_builder.build_cot_prompt(query, final_context_chunks)
        raw_output = await self.llm_client.generate_async(prompt)
        
        # 4. Parse & Validate Safety
        try:
            validated_response = self.response_parser.parse(raw_output)
            
            # 5. Output Guard (Phase 6 - Grounding)
            if settings.ENABLE_SAFETY_GUARDRAILS and self.output_guard:
                grounding_score = self.output_guard.check_grounding(validated_response.answer, final_context_chunks)
                if grounding_score < settings.GROUNDING_SIMILARITY_THRESHOLD:
                    logger.warning(f"Response failed grounding check (Score: {grounding_score:.2f})")
                    from evaluation.safety_monitor import safety_monitor
                    safety_monitor.log_violation("grounding_failure", validated_response.answer, grounding_score, details="Semantic drift detected")
                    validated_response.answer = "I'm sorry, but I couldn't find sufficient grounded evidence in my knowledge base to answer that confidently."
                    validated_response.confidence = 0.0
            
            return validated_response
        except Exception as e:
            logger.error(f"Generation orchestration failed: {e}")
            raise
