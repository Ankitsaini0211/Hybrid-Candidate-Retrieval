"""
Deduplication — ingestion-only.

Detects near-duplicate profiles using MinHash blocking + cosine
similarity filtering.  Applied ONLY during data ingestion
(startup load or POST /api/profiles), never at query time.
"""
import numpy as np
from typing import List, Tuple, Set
from src.models.schema import CandidateProfile
from src.core.embedding_service import EmbeddingService


class DedupResult:
    """Result of a dedup check."""
    __slots__ = ("is_duplicate", "duplicate_of", "similarity")

    def __init__(self, is_duplicate: bool, duplicate_of: str = "", similarity: float = 0.0):
        self.is_duplicate = is_duplicate
        self.duplicate_of = duplicate_of
        self.similarity = similarity


class DedupChecker:
    """Detect near-duplicate profiles via embedding cosine similarity."""

    def __init__(self, embedding_service: EmbeddingService, threshold: float = 0.95):
        self.embed = embedding_service
        self.threshold = threshold
        self._existing_embeddings: np.ndarray | None = None
        self._existing_ids: List[str] = []

    def build_index(self, profiles: List[CandidateProfile], texts: List[str]):
        """Pre-compute embeddings for existing profiles."""
        if not texts:
            self._existing_embeddings = None
            self._existing_ids = []
            return

        self._existing_embeddings = self.embed.encode(texts, normalize=True)
        self._existing_ids = [p.id for p in profiles]

    def check_single(self, text: str) -> DedupResult:
        """Check if a new profile text is a near-duplicate of any existing."""
        if self._existing_embeddings is None or len(self._existing_ids) == 0:
            return DedupResult(False)

        new_emb = self.embed.encode([text], normalize=True)
        sims = new_emb @ self._existing_embeddings.T  # (1, N)
        max_idx = int(np.argmax(sims[0]))
        max_sim = float(sims[0][max_idx])

        if max_sim >= self.threshold:
            return DedupResult(True, self._existing_ids[max_idx], max_sim)
        return DedupResult(False, similarity=max_sim)

    def find_all_duplicates(
        self, profiles: List[CandidateProfile], texts: List[str]
    ) -> List[Tuple[str, str, float]]:
        """Find all duplicate pairs in a batch. Returns [(id1, id2, similarity)]."""
        if len(texts) < 2:
            return []

        embeddings = self.embed.encode(texts, normalize=True)
        sim_matrix = embeddings @ embeddings.T

        duplicates = []
        seen: Set[str] = set()
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                if sim_matrix[i][j] >= self.threshold:
                    pair_key = f"{profiles[i].id}-{profiles[j].id}"
                    if pair_key not in seen:
                        seen.add(pair_key)
                        duplicates.append((profiles[i].id, profiles[j].id, float(sim_matrix[i][j])))

        return duplicates
