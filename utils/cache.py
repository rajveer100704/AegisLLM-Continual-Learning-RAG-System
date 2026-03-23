import redis
import json
import pickle
from typing import Optional, Any
from configs.config import settings
from utils.logger import logger

class RedisCache:
    """
    Simple Redis-based caching layer for embeddings and LLM responses.
    """
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=0,
                decode_responses=False # Store bytes for pickle/json
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis cache at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis. Caching will be disabled. Error: {e}")
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        if not self.client: return None
        data = self.client.get(key)
        if data:
            try:
                return pickle.loads(data)
            except:
                return data.decode("utf-8")
        return None

    def set(self, key: str, value: Any, expire: int = 3600):
        if not self.client: return
        try:
            if isinstance(value, (dict, list, str, int, float)):
                data = pickle.dumps(value)
            else:
                data = value
            self.client.setex(key, expire, data)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")

cache = RedisCache()
