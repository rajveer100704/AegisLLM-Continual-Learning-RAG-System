import time
import asyncio
import numpy as np
import json
from typing import List, Dict, Any
from pipeline.rag_pipeline import RAGPipeline
from evaluation.benchmark_data import BENCHMARK_DATA
from utils.logger import logger
from configs.config import settings

class ResearchAblationEngine:
    """
    Research-Grade Ablation: Statistical reliability, controlled variables, and diverse metrics.
    """
    def __init__(self, iterations: int = 3):
        self.pipeline = RAGPipeline()
        self.iterations = iterations
        self.raw_results = []

    async def run_controlled_experiment(self, config_name: str, settings_override: Dict[str, Any], query_obj: Dict[str, Any]):
        """
        Runs a single configuration N times and returns stats.
        """
        latencies = []
        recalls = []
        token_counts = []
        
        query = query_obj["query"]
        gt_ids = query_obj["ground_truth_docs"]

        for i in range(self.iterations):
            start_time = time.time()
            
            # Simulated configuration toggling logic
            # In a real system, we would use dependency injection or feature flags
            if config_name == "Dense Baseline":
                hits = self.pipeline.dense_retriever.retrieve(query)
            elif config_name == "Hybrid (Dense+Sparse)":
                hits = self.pipeline.hybrid_retriever.retrieve(query)
            elif config_name == "Hybrid + Rerank":
                hits = self.pipeline.hybrid_retriever.retrieve(query)
                hits = self.pipeline.reranker.rerank(query, hits)
            elif config_name == "Elite (All Optimized)":
                # Full pipeline flow
                rewritten = await self.pipeline.rewriter.rewrite(query)
                hits = self.pipeline.hybrid_retriever.retrieve(rewritten)
                hits = self.pipeline.reranker.rerank(rewritten, hits)
                hits = self.pipeline.generator.compressor.compress(rewritten, hits)
                
            latency = (time.time() - start_time) * 1000
            
            # Metrics
            found_ids = [h.get("doc_id") for h in hits]
            recall = len(set(found_ids) & set(gt_ids)) / len(gt_ids) if gt_ids else 0.0
            tokens = sum(len(h.get("content", "")) for h in hits) // 4
            
            latencies.append(latency)
            recalls.append(recall)
            token_counts.append(tokens)

        return {
            "config": config_name,
            "query_id": query_obj["id"],
            "recall_mean": np.mean(recalls),
            "recall_std": np.std(recalls),
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "tokens_avg": int(np.mean(token_counts)),
            "cost_est": np.mean(token_counts) * (1.25 / 1_000_000) # Simple input-only estimate
        }

    async def run_all(self):
        configs = ["Dense Baseline", "Hybrid (Dense+Sparse)", "Hybrid + Rerank", "Elite (All Optimized)"]
        
        for query_obj in BENCHMARK_DATA:
            logger.info(f"🧐 Evaluating Query: {query_obj['id']} ({query_obj['type']})")
            for cfg in configs:
                res = await self.run_controlled_experiment(cfg, {}, query_obj)
                self.raw_results.append(res)
                
        self.generate_final_report()

    def generate_final_report(self):
        logger.info("📄 Generating Research-Grade Evaluation Report...")
        # (Simplified print, would write to evaluation_results.md in real use)
        print("\n" + "="*80)
        print(f"{'Config':<25} | {'Recall':<15} | {'Latency (P95)':<15} | {'Avg Tokens':<10}")
        print("-" * 80)
        
        # Aggregate by config
        for cfg in ["Dense Baseline", "Hybrid (Dense+Sparse)", "Hybrid + Rerank", "Elite (All Optimized)"]:
            cfg_results = [r for r in self.raw_results if r["config"] == cfg]
            r_mean = np.mean([r["recall_mean"] for r in cfg_results])
            r_std = np.mean([r["recall_std"] for r in cfg_results])
            l_p95 = np.mean([r["latency_p95"] for r in cfg_results])
            t_avg = np.mean([r["tokens_avg"] for r in cfg_results])
            
            print(f"{cfg:<25} | {r_mean:.2f} ± {r_std:.2f} | {l_p95:.2f}ms | {int(t_avg):<10}")

if __name__ == "__main__":
    engine = ResearchAblationEngine()
    asyncio.run(engine.run_all())
