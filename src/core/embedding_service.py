"""
Centralized Embedding Service.

Holds a *single* SentenceTransformer instance shared by every component
(vector indexer, graph feature extractor, reranker prep, dedup, etc.).
Avoids duplicate model loading and provides a simple caching layer.
"""
import numpy as np
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Singleton-style embedding provider for the entire pipeline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[EmbeddingService] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        self._cache: Dict[str, np.ndarray] = {}
        print(f"[EmbeddingService] Ready — dimension={self.dimension}")

    # ------------------------------------------------------------------
    # Core encode
    # ------------------------------------------------------------------
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: raw strings to embed.
            normalize: L2-normalize for cosine similarity via dot product.
            show_progress: show tqdm bar for large batches.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        if normalize:
            import faiss
            faiss.normalize_L2(embeddings)
        return embeddings

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Convenience: encode one text, return 1-D vector."""
        return self.encode([text], normalize=normalize)[0]

    # ------------------------------------------------------------------
    # Cached encode (for repeated lookups like graph node names)
    # ------------------------------------------------------------------
    def encode_cached(self, text: str, normalize: bool = True) -> np.ndarray:
        """Return cached embedding if available, else compute and cache."""
        if text in self._cache:
            return self._cache[text]
        vec = self.encode_single(text, normalize=normalize)
        self._cache[text] = vec
        return vec

    def clear_cache(self):
        self._cache.clear()

    # ------------------------------------------------------------------
    # Batch similarity (useful for dedup, graph feature extraction)
    # ------------------------------------------------------------------
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors."""
        return float(np.dot(vec_a, vec_b))

    def pairwise_cosine(self, embeddings: np.ndarray) -> np.ndarray:
        """Return NxN cosine similarity matrix (assumes L2-normalized input)."""
        return embeddings @ embeddings.T
