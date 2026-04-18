"""
ColBERT Re-ranker (Stage 2).

Late-interaction reranking: encode query and document tokens independently,
then compute MaxSim (maximum similarity between each query token and all
document tokens). This approximates cross-encoder accuracy at ~100× lower
cost because document token embeddings are precomputed.

When ColBERT model is not available, falls back to a lightweight
embedding-based reranker using the shared EmbeddingService.
"""
import numpy as np
from typing import List, Dict, Any, Protocol, Optional
from src.core.embedding_service import EmbeddingService
from src.core.feature_pipeline import FeaturePipeline
from src.models.schema import CandidateProfile


class Reranker(Protocol):
    """Protocol for any reranking strategy."""
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]: ...


class NoopReranker:
    """Pass-through — returns candidates unchanged."""

    model_name = "none"

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return candidates[:top_k]


class ColBERTReranker:
    """
    Late-interaction reranker using MaxSim scoring.

    Approximates ColBERT by:
    1. Tokenizing query and document into word-level chunks
    2. Encoding each chunk with the shared bi-encoder
    3. Computing MaxSim: for each query token, find the max cosine
       similarity across all document tokens, then average.

    This is a simplified ColBERT-style approach that works with any
    SentenceTransformer model — no dedicated ColBERT model needed.
    """

    model_name = "colbert-style-maxsim"

    def __init__(
        self,
        embedding_service: EmbeddingService,
        feature_pipeline: FeaturePipeline,
    ):
        self.embed = embedding_service
        self.pipeline = feature_pipeline
        print("[ColBERTReranker] Initialized (MaxSim over token embeddings).")

    def _tokenize_to_chunks(self, text: str, max_chunks: int = 32) -> List[str]:
        """Split text into meaningful chunks for token-level comparison."""
        words = text.lower().split()
        if len(words) <= max_chunks:
            return words if words else [text]

        # Group into chunks of ~3 words each
        chunk_size = max(1, len(words) // max_chunks)
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks[:max_chunks]

    def _maxsim_score(self, query_embeds: np.ndarray, doc_embeds: np.ndarray) -> float:
        """Compute MaxSim: avg of max-cosine per query token."""
        if query_embeds.shape[0] == 0 or doc_embeds.shape[0] == 0:
            return 0.0

        # Similarity matrix: (num_query_tokens, num_doc_tokens)
        sim_matrix = query_embeds @ doc_embeds.T

        # For each query token, take the max similarity across all doc tokens
        max_sims = np.max(sim_matrix, axis=1)  # shape: (num_query_tokens,)

        return float(np.mean(max_sims))

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # Tokenize query into chunks
        query_chunks = self._tokenize_to_chunks(query, max_chunks=16)
        query_embeds = self.embed.encode(query_chunks, normalize=True)

        # Score each candidate
        for cand in candidates:
            profile: CandidateProfile = cand["profile"]
            doc_text = self.pipeline.build_text(profile)
            doc_chunks = self._tokenize_to_chunks(doc_text, max_chunks=32)
            doc_embeds = self.embed.encode(doc_chunks, normalize=True)

            colbert_score = self._maxsim_score(query_embeds, doc_embeds)
            cand["colbert_score"] = colbert_score

        # Sort by ColBERT score (descending)
        candidates.sort(key=lambda c: c.get("colbert_score", 0.0), reverse=True)

        return candidates[:top_k]
