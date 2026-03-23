import numpy as np
from typing import List, Optional
from utils.logger import logger
from configs.config import settings

class DriftMonitor:
    """
    Monitors semantic drift by tracking similarity distribution changes.
    """
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.15):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.similarity_history: List[float] = []
        self.baseline_similarity: Optional[float] = None

    def add_sample(self, avg_similarity: float):
        """Adds a new similarity sample and checks for drift."""
        self.similarity_history.append(avg_similarity)
        if len(self.similarity_history) > self.window_size:
            self.similarity_history.pop(0)
            
        # Initialize baseline
        if self.baseline_similarity is None and len(self.similarity_history) >= 20:
            self.baseline_similarity = np.mean(self.similarity_history)
            logger.info(f"DriftMonitor: Baseline search similarity established at {self.baseline_similarity:.4f}")
            
        # Check for drift
        if self.baseline_similarity is not None:
            current_avg = np.mean(self.similarity_history)
            drift = abs(current_avg - self.baseline_similarity)
            
            if drift > self.drift_threshold:
                logger.warning(
                    f"⚠️ SEMANTIC DRIFT DETECTED: {drift:.4f} > {self.drift_threshold}. "
                    f"Baseline: {self.baseline_similarity:.4f} | Current: {current_avg:.4f}"
                )
                self._trigger_alert()

    def _trigger_alert(self):
        """Placeholder for alerting logic (e.g. trigger re-embedding)."""
        logger.error("DRIFT ALERT: System performance may be degrading. Consider index maintenance.")

drift_monitor = DriftMonitor()
