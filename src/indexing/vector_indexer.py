"""
HNSW-backed Vector Indexer.

Replaces the original brute-force ``IndexFlatL2`` with
``IndexHNSWFlat`` wrapped in ``IndexIDMap2`` for:
  - O(log n) approximate nearest-neighbor search
  - Incremental add / remove by numeric ID
  - Configurable speed-vs-accuracy via efSearch

Consumes ``EmbeddingService`` (shared model) and ``FeaturePipeline``
(modular text construction) — no duplicate logic.
"""
import faiss
import numpy as np
from typing import List, Dict, Any, Optional

from src.models.schema import CandidateProfile
from src.core.embedding_service import EmbeddingService
from src.core.feature_pipeline import FeaturePipeline
from src.core.config import SystemConfig


class VectorIndexer:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        feature_pipeline: FeaturePipeline,
        config: SystemConfig | None = None,
    ):
        self.embed = embedding_service
        self.pipeline = feature_pipeline
        self.config = config or SystemConfig()

        # Build the FAISS index
        dim = self.embed.dimension
        if self.config.index_type == "hnsw":
            base_index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
            base_index.hnsw.efConstruction = self.config.hnsw_ef_construction
            base_index.hnsw.efSearch = self.config.hnsw_ef_search
        elif self.config.index_type == "flat":
            base_index = faiss.IndexFlatIP(dim)  # inner-product on L2-normed = cosine
        else:
            # Default fallback to flat
            base_index = faiss.IndexFlatIP(dim)

        # Wrap with IDMap2 so we can remove by ID
        self.index = faiss.IndexIDMap2(base_index)

        # Bookkeeping
        self.profiles: List[CandidateProfile] = []
        self._id_to_faiss_id: Dict[str, int] = {}   # profile.id → numeric faiss id
        self._faiss_id_to_profile: Dict[int, CandidateProfile] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Bulk index (startup)
    # ------------------------------------------------------------------
    def index_all(self, profiles: List[CandidateProfile]):
        """Build index from scratch for all profiles."""
        self.profiles = profiles
        if not profiles:
            return

        texts = self.pipeline.build_texts(profiles)
        print(f"[VectorIndexer] Embedding {len(texts)} documents ({self.config.index_type})...")
        embeddings = self.embed.encode(texts, normalize=True)

        # Assign sequential numeric IDs
        ids = np.arange(len(profiles), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        for i, p in enumerate(profiles):
            self._id_to_faiss_id[p.id] = i
            self._faiss_id_to_profile[i] = p
        self._next_id = len(profiles)

        print(f"[VectorIndexer] Indexed {self.index.ntotal} vectors.")

    # ------------------------------------------------------------------
    # Incremental add
    # ------------------------------------------------------------------
    def add_single(self, profile: CandidateProfile):
        """Add one profile to the live index."""
        text = self.pipeline.build_text(profile)
        embedding = self.embed.encode([text], normalize=True)

        fid = self._next_id
        self._next_id += 1

        self.index.add_with_ids(embedding, np.array([fid], dtype=np.int64))
        self._id_to_faiss_id[profile.id] = fid
        self._faiss_id_to_profile[fid] = profile
        self.profiles.append(profile)

    # ------------------------------------------------------------------
    # Incremental remove
    # ------------------------------------------------------------------
    def remove(self, profile_id: str) -> bool:
        """Remove a profile by its string ID. Returns True if found."""
        fid = self._id_to_faiss_id.get(profile_id)
        if fid is None:
            return False

        self.index.remove_ids(np.array([fid], dtype=np.int64))
        del self._id_to_faiss_id[profile_id]
        del self._faiss_id_to_profile[fid]
        self.profiles = [p for p in self.profiles if p.id != profile_id]
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic nearest-neighbor search.

        Returns list of {profile, score, technique} dicts sorted by score desc.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embed.encode([query], normalize=True)

        # Inner product on L2-normalized vectors = cosine similarity
        scores, ids = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for i, fid in enumerate(ids[0]):
            if fid == -1:
                continue
            profile = self._faiss_id_to_profile.get(int(fid))
            if profile is None:
                continue
            results.append({
                "profile": profile,
                "score": float(scores[0][i]),
                "technique": "faiss",
            })
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": self.index.ntotal,
            "index_type": self.config.index_type,
            "dimension": self.embed.dimension,
            "hnsw_m": self.config.hnsw_m,
            "hnsw_ef_search": self.config.hnsw_ef_search,
        }
