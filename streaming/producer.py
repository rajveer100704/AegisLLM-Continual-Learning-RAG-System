import redis
import json
from typing import Dict, Any
from configs.config import settings
from streaming.schemas import IngestionEvent
from utils.logger import logger, observe_latency

class StreamProducer:
    """
    Produces validated ingestion events to Redis Streams.
    """
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )
            self.stream_name = settings.REDIS_STREAM_NAME
            logger.info(f"StreamProducer connected to Redis stream: {self.stream_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for producing: {e}")
            self.client = None

    @observe_latency
    def produce(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Validates and pushes an event to the stream."""
        if not self.client: return None
        
        try:
            event = IngestionEvent(
                doc_id=doc_id,
                content=content,
                metadata=metadata or {}
            )
            
            event_id = self.client.xadd(
                self.stream_name,
                event.to_redis_dict()
            )
            
            logger.info(f"Event {event.event_id} produced to stream. Redis ID: {event_id}")
            return event_id
        except Exception as e:
            logger.error(f"Error producing event: {e}")
            return None

# Singleton instance
producer = StreamProducer()
