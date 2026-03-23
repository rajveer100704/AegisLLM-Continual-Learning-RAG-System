import json
from typing import Dict, Any
from utils.logger import logger
from configs.config import settings

class CostTracker:
    """
    Elite Optimization: Track token usage and estimate API costs.
    Pricing (Gemini 2.5 Pro Approx): $1.25 / 1M input, $5.0 / 1M output tokens.
    """
    def __init__(self):
        self.stats_file = settings.DATA_DIR / "cost_stats.json"
        self._load_stats()
        self.PRICING = {
            "input": 1.25 / 1_000_000,
            "output": 5.0 / 1_000_000
        }

    def _load_stats(self):
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {"total_input": 0, "total_output": 0, "total_cost": 0.0}

    def log_call(self, input_tokens: int, output_tokens: int):
        """Update cumulative stats."""
        cost = (input_tokens * self.PRICING["input"]) + (output_tokens * self.PRICING["output"])
        self.stats["total_input"] += input_tokens
        self.stats["total_output"] += output_tokens
        self.stats["total_cost"] += cost
        
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f)
        
        logger.info(f"Query Cost: ${cost:.6f} | Cumulative: ${self.stats['total_cost']:.4f}")

    def get_summary(self) -> Dict[str, Any]:
        return self.stats

cost_tracker = CostTracker()
