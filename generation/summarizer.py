from typing import List, Dict, Any
from generation.llm_client import LLMClient
from utils.logger import logger, observe_latency

class MapReduceSummarizer:
    """
    Map-Reduce pipeline for synthesising large contexts into a reasoning-optimized summary.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    @observe_latency
    async def summarize_recursive(self, query: str, contents: List[str]) -> str:
        """
        Recursively summarize chunks in pairs until a final summary is reached.
        """
        if not contents: return ""
        if len(contents) == 1: return contents[0]

        logger.info(f"Map-Reduce Step: Summarizing {len(contents)} chunks...")
        
        # 1. Map: Summarize in batches
        summaries = []
        batch_size = 3
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i+batch_size]
            prompt = self._build_map_prompt(query, batch)
            summary = await self.llm.generate_async(prompt)
            summaries.append(summary)

        # 2. Reduce (Recursion)
        return await self.summarize_recursive(query, summaries)

    def _build_map_prompt(self, query: str, context_list: List[str]) -> str:
        context_text = "\n---\n".join(context_list)
        return f"""
        Summarize the following context blocks specifically as they relate to the question: "{query}"
        Focus on extracting distinct facts and resolving contradictions.

        CONTEXT:
        {context_text}

        SUMMARY (JSON output only - "summary"):
        """
