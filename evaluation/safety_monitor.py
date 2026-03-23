import json
from pathlib import Path
from datetime import datetime
from utils.logger import logger

class SafetyMonitor:
    """
    Logs safety violations for auditing and governance.
    """
    def __init__(self, log_dir: str = "data/safety_logs"):
        self.log_path = Path(log_dir)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path / "violations.jsonl"

    def log_violation(self, event_type: str, query: str, score: float, details: str = ""):
        """Appends a safety violation to the JSONL log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "query": query,
            "score": round(score, 3),
            "details": details
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"🛡️ Safety violation logged: {event_type}")
        except Exception as e:
            logger.error(f"Failed to log safety violation: {e}")

safety_monitor = SafetyMonitor()
