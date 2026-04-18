"""
Central system configuration.

All feature flags, index parameters, and scoring defaults live here.
Every component reads from SystemConfig — no hardcoded magic numbers.
"""
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DynamicWeights:
    """Per-signal weights used in both RRF fusion and final scoring."""
    bm25: float = 1.0
    faiss: float = 1.0
    graph: float = 1.0
    colbert: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {"bm25": self.bm25, "faiss": self.faiss, "graph": self.graph, "colbert": self.colbert}


@dataclass
class SystemConfig:
    """Master configuration — consumed by every pipeline component."""

    # ── Feature Flags ──────────────────────────────────────────────────
    enable_colbert_rerank: bool = True
    enable_cross_encoder: bool = False    # debug only, never in pipeline
    enable_template_answers: bool = True
    dynamic_weighting: bool = True
    query_adaptive: bool = True

    # ── Index Parameters ───────────────────────────────────────────────
    index_type: str = "hnsw"              # "hnsw" | "ivfpq" | "flat"
    hnsw_m: int = 32                      # HNSW connections per node
    hnsw_ef_construction: int = 200       # HNSW build-time quality
    hnsw_ef_search: int = 128             # HNSW query-time quality
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval Parameters ───────────────────────────────────────────
    rrf_k: int = 60                       # RRF smoothing constant
    top_k_for_rerank: int = 100           # candidates sent to Stage 2
    threshold_ratio: float = 0.15         # min score as fraction of top
    threshold_floor: float = 0.02         # absolute minimum score

    # ── Dedup Parameters ───────────────────────────────────────────────
    dedup_threshold: float = 0.95         # cosine sim threshold for dedup
    dedup_enabled: bool = True

    # ── Default Weights (used when dynamic_weighting is off) ───────────
    default_weights: DynamicWeights = field(default_factory=DynamicWeights)

    # ── Neo4j Connection (reads from environment — matches Aura variable names)
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j")))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))
    aura_instance_id: str = field(default_factory=lambda: os.getenv("AURA_INSTANCEID", ""))
