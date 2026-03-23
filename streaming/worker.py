import asyncio
from pipeline.rag_pipeline import RAGPipeline
from streaming.consumer import StreamConsumer
from streaming.schemas import IngestionEvent
from utils.logger import logger

class AegisWorker:
    """
    Orchestrates the link between Redis Streams and the RAG Pipeline.
    Handles ingestion, indexing, and persistent snapshots.
    """
    def __init__(self):
        self.pipeline = RAGPipeline()
        self.consumer = StreamConsumer(consumer_name="aegis_worker_01")
        self.batch_counter = 0

    async def _handle_ingestion(self, event: IngestionEvent):
        """Callback for the stream consumer to process an ingestion event."""
        logger.info(f"Worker processing event: {event.event_id} from {event.doc_id}")
        
        # 1. Pipeline Ingestion (Chunk -> Embed -> Index)
        await self.pipeline.ingest_document(event.content, source=event.doc_id)
        
        # 2. Snapshot management
        self.batch_counter += 1
        if self.batch_counter >= 10:
            logger.info("Triggering periodic Index Snapshot...")
            self.pipeline.vector_store.save()
            self.batch_counter = 0

    async def start(self):
        """Main entry point for the worker."""
        logger.info("AegisLLM Streaming Worker Online.")
        await self.consumer.run(self._handle_ingestion)

if __name__ == "__main__":
    worker = AegisWorker()
    asyncio.run(worker.start())
