import time
import functools
from loguru import logger
import sys
from configs.config import settings

# Configure Loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL
)

def observe_latency(func):
    """Decorator to measure and log function latency."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency = end_time - start_time
        logger.info(f"Latency [{func.__name__}]: {latency:.4f}s")
        return result
    return wrapper

def observe_latency_async(func):
    """Decorator to measure and log async function latency."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        latency = end_time - start_time
        logger.info(f"Latency [{func.__name__}]: {latency:.4f}s")
        return result
    return wrapper
