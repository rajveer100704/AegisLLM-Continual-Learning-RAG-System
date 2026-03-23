import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from configs.config import settings
from utils.logger import logger, observe_latency

class EmbeddingClient:
    """
    Wrapper for SentenceTransformers to generate dense embeddings.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded embedding model: {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    @observe_latency
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for a single text or a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                normalize_embeddings=True # Score normalization prep
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
