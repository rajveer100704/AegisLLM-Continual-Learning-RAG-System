import uuid
import datetime
from typing import List, Dict, Any
from configs.config import settings
from utils.logger import observe_latency

class TextChunker:
    """
    Advanced text chunker with token-aware splitting and metadata enrichment.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    @observe_latency
    def chunk_text(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks and attach metadata.
        NOTE: In a real production system, you'd use tiktoken or similar for exact token counts.
        For this implementation, we use a character-based proxy (4 chars ~ 1 token) 
        to keep dependencies minimal while maintaining the logic.
        """
        # Character-based proxy for tokens
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        
        chunks = []
        doc_id = str(uuid.uuid4())
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + char_chunk_size
            chunk_content = text[start:end]
            
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_idx}",
                "source": source,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "chunk_index": chunk_idx,
                "content": chunk_content
            }
            
            chunks.append(chunk_metadata)
            
            if end >= len(text):
                break
                
            start += (char_chunk_size - char_overlap)
            chunk_idx += 1
            
        return chunks
