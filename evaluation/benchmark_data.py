from typing import List, Dict, Any

BENCHMARK_DATA = [
    {
        "id": "q1_factual",
        "type": "factual",
        "query": "What are the core technical components of AegisLLM?",
        "ground_truth_docs": ["news_01"], # Hypothetical doc_id
        "description": "Standard technical retrieval."
    },
    {
        "id": "q2_ambiguous",
        "type": "ambiguous",
        "query": "How does it handle things?",
        "ground_truth_docs": ["news_01"],
        "description": "Tests Query Rewriting efficacy."
    },
    {
        "id": "q3_temporal",
        "type": "temporal",
        "query": "What is the latest update as of today?",
        "ground_truth_docs": ["news_01"],
        "description": "Tests Temporal Decay scoring."
    },
    {
        "id": "q4_redundant",
        "type": "redundant",
        "query": "Tell me about AegisLLM features repeatedly.",
        "ground_truth_docs": ["news_01"],
        "description": "Tests Context Compression token savings."
    }
]
