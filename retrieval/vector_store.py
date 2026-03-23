import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from configs.config import settings
from utils.logger import logger, observe_latency

class VectorStore:
    """
    Metadata-aware vector store using FAISS and a local ID-to-metadata mapping.
    """
    def __init__(self, dimension: int = None, index_path: str = None):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_path = index_path or settings.INDEX_PATH
        self.metadata_path = f"{self.index_path}.metadata.pkl"
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension) # Inner product for cosine similarity with normalized vectors
        self.index = faiss.IndexIDMap(self.index)
        
        # Metadata storage
        self.metadata_map: Dict[int, Dict[str, Any]] = {}
        self._current_id = 0
        
        if os.path.exists(f"{self.index_path}.index"):
            self.load()

    @observe_latency
    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Add embeddings and their associated metadata with temporal tagging."""
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadatas must have the same length.")
        
        num_new = len(embeddings)
        ids = np.array(range(self._current_id, self._current_id + num_new), dtype=np.int64)
        
        self.index.add_with_ids(embeddings.astype("float32"), ids)
        
        import datetime
        for i, meta in enumerate(metadatas):
            # Ensure ingestion timestamp for Phase 3 temporal scoring
            if "ingestion_timestamp" not in meta:
                meta["ingestion_timestamp"] = datetime.datetime.utcnow().isoformat()
            self.metadata_map[self._current_id + i] = meta
            
        self._current_id += num_new
        logger.info(f"Added {num_new} vectors to index. Total: {self.index.ntotal}")

    @observe_latency
    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for the most similar vectors and return their metadata."""
        top_k = top_k or settings.TOP_K
        
        distances, ids = self.index.search(query_embedding.astype("float32"), top_k)
        
        results = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1: continue
            
            meta = self.metadata_map.get(int(idx), {}).copy()
            meta["score"] = float(dist)
            results.append(meta)
            
        return results

    def save(self):
        """Persist the index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.index")
        with open(self.metadata_path, "wb") as f:
            pickle.dump({"metadata": self.metadata_map, "current_id": self._current_id}, f)
        logger.info(f"Saved index and metadata to {self.index_path}")

    def load(self):
        """Load the index and metadata from disk."""
        if os.path.exists(f"{self.index_path}.index"):
            self.index = faiss.read_index(f"{self.index_path}.index")
            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
                self.metadata_map = data["metadata"]
                self._current_id = data["current_id"]
            logger.info(f"Loaded index and metadata. Total vectors: {self.index.ntotal}")
