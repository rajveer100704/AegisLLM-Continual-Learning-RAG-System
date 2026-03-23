from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class IngestionEvent(BaseModel):
    """
    Strict schema for streaming ingestion events.
    Ensures idempotency and auditability.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    operation: str = "UPSERT" # UPSERT or DELETE
    version: int = 1

    def to_redis_dict(self) -> Dict[str, str]:
        """Convert to flat dictionary for Redis Streams."""
        import json
        return {
            "event_id": self.event_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": json.dumps(self.metadata),
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "version": str(self.version)
        }

    @classmethod
    def from_redis_dict(cls, data: Dict[bytes, bytes]):
        """Parse from Redis stream payload."""
        import json
        # Decode bytes if necessary
        d = {k.decode("utf-8") if isinstance(k, bytes) else k: 
             v.decode("utf-8") if isinstance(v, bytes) else v 
             for k, v in data.items()}
        
        return cls(
            event_id=d["event_id"],
            doc_id=d["doc_id"],
            content=d["content"],
            metadata=json.loads(d["metadata"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            operation=d["operation"],
            version=int(d["version"])
        )
