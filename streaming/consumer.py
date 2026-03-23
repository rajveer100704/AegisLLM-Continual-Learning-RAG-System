import redis
import time
import asyncio
from typing import Dict, Any, Callable
from configs.config import settings
from streaming.schemas import IngestionEvent
from utils.logger import logger, observe_latency

class StreamConsumer:
    """
    Industrial-grade consumer using Redis Consumer Groups for fault tolerance.
    """
    def __init__(self, consumer_name: str = "main_worker"):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=False # Keep raw for schema parsing
        )
        self.stream_name = settings.REDIS_STREAM_NAME
        self.group_name = settings.REDIS_CONSUMER_GROUP
        self.consumer_name = consumer_name
        self._setup_group()

    def _setup_group(self):
        """Create consumer group if it doesn't exist."""
        try:
            self.client.xgroup_create(self.stream_name, self.group_name, id="0", mkstream=True)
            logger.info(f"Consumer group {self.group_name} created.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group {self.group_name} already exists.")
            else:
                raise

    async def run(self, process_callback: Callable[[IngestionEvent], Any]):
        """
        Polls the stream using XREADGROUP and processes events with ACK.
        """
        logger.info(f"Consumer {self.consumer_name} starting...")
        
        while True:
            try:
                # Read from group
                # ">" means read new messages never delivered to others
                messages = self.client.xreadgroup(
                    self.group_name, 
                    self.consumer_name, 
                    {self.stream_name: ">"}, 
                    count=1, 
                    block=5000
                )
                
                if not messages:
                    continue

                for stream_name, msg_list in messages:
                    for redis_id, payload in msg_list:
                        try:
                            # 1. Parse Event
                            event = IngestionEvent.from_redis_dict(payload)
                            logger.info(f"Processing event {event.event_id} (ID: {redis_id})")
                            
                            # 2. Execute business logic (Indexing)
                            await process_callback(event)
                            
                            # 3. Acknowledge (ACK)
                            self.client.xack(self.stream_name, self.group_name, redis_id)
                            logger.debug(f"Event {event.event_id} ACKed.")
                            
                        except Exception as e:
                            logger.error(f"Failed to process message {redis_id}: {e}")
                            # In production, push to DLQ here
                            
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Stream consumer error: {e}")
                await asyncio.sleep(5) # Backoff
