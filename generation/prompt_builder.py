from typing import List, Dict, Any
import datetime

class PromptBuilder:
    """
    Constructs structured prompts for Gemini to ensure grounding and JSON output.
    """
    SYSTEM_INSTRUCTION = """
    You are AegisLLM, a production-grade RAG system logic unit.
    Your mission is to provide accurate, grounded, and concise answers based ONLY on the provided context.
    
    STRICT RULES:
    1. Only use the provided context. If the answer is not in the context, say "I don't have enough information".
    2. Do NOT hallucinate.
    3. Always list your sources (chunk IDs) used.
    4. Provide a confidence score (0.0 to 1.0) based on how well the context covers the query.
    5. You MUST output in JSON format.
    """

    @staticmethod
    def build_cot_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Constructs a Chain-of-Thought reasoning prompt with Instruction Isolation.
        """
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            context_str += f"[DOC {chunk['chunk_id']}]\n{chunk['content']}\n\n"

        return f"""
<SYSTEM_INSTRUCTIONS>
{PromptBuilder.SYSTEM_INSTRUCTION}
Follow these steps EXACTLY:
1. EXAMINE: What is the user actually asking?
2. EVIDENCE: Which context blocks contain the direct answer?
3. SYNTHESIZE: Formulate a grounded response with citations.
</SYSTEM_INSTRUCTIONS>

<CONTEXT_KNOWLEDGE_BASE>
{context_str}
</CONTEXT_KNOWLEDGE_BASE>

<USER_QUERY_DATA_ONLY>
{query}
</USER_QUERY_DATA_ONLY>

JSON OUTPUT SCHEMA:
{{
    "reasoning": "Step-by-step logic here",
    "answer": "Grounded answer",
    "sources": ["ids"],
    "confidence": 0.9
}}
"""
