"""
Evaluation Script — Updated for the Dynamic Semantic Retrieval pipeline.

Runs ablation tests across individual components and the full pipeline:
  - BM25 only
  - HNSW only
  - BM25 + HNSW (RRF)
  - Full pipeline (+ Graph Features + ColBERT)

Reports: P@k, R@k, nDCG@k, MRR per configuration.
"""
import os
import math
import time
from typing import List, Dict

from src.core.config import SystemConfig
from src.core.data_loader import load_and_clean_data
from src.core.embedding_service import EmbeddingService
from src.core.feature_pipeline import FeaturePipeline
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.vector_indexer import VectorIndexer
from src.indexing.graph_indexer import GraphIndexer
from src.core.graph_features import GraphFeatureExtractor
from src.core.reranker import ColBERTReranker, NoopReranker
from src.core.retriever import HybridRetriever


# ── IR Metrics ─────────────────────────────────────────────────────────
def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_retrieved / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
    return relevant_retrieved / len(relevant)


def dcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            rel = 1
            dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved, relevant, k):
    idcg_docs = list(relevant)
    idcg = dcg_at_k(idcg_docs, relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


def mrr(retrieved, relevant):
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ── Evaluation ─────────────────────────────────────────────────────────
def evaluate_system():
    print("=" * 60)
    print("  Dynamic Semantic Retrieval — Evaluation Suite")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "data", "profiles.csv")
    profiles = load_and_clean_data(data_path)
    print(f"Loaded {len(profiles)} profiles.\n")

    # ── Shared services ────────────────────────────────────────────────
    config = SystemConfig()
    embed = EmbeddingService(config.embedding_model)
    pipeline = FeaturePipeline()

    # ── Build indexes ──────────────────────────────────────────────────
    bm25 = BM25Indexer()
    bm25.set_pipeline(pipeline)
    bm25.index(profiles)

    vector = VectorIndexer(embed, pipeline, config)
    vector.index_all(profiles)

    # Graph (optional)
    graph_features = None
    try:
        graph = GraphIndexer()
        graph.index(profiles)
        graph_features = GraphFeatureExtractor(graph)
    except Exception as e:
        print(f"[WARNING] Neo4j unavailable: {e}. Skipping graph features.\n")

    # ColBERT
    colbert = ColBERTReranker(embed, pipeline)

    # ── Test queries ───────────────────────────────────────────────────
    test_queries = [
        ("Looking for a Python Backend Developer", ["181457", "232193", "231910"]),
        ("Regulatory Affairs Manager with FDA experience", ["180869", "230818", "232500"]),
        ("Machine Learning Engineer with 3 years experience", ["232936", "231910", "232193"]),
        ("Full Stack Developer React Node.js", ["232936", "181457", "231910"]),
        ("Data Scientist with Python and SQL skills", ["232193", "232936", "181457"]),
    ]

    k = 5

    # ── Ablation configs ───────────────────────────────────────────────
    configs = {
        "BM25 Only": {"colbert": False, "graph": False, "vector": False},
        "HNSW Only": {"colbert": False, "graph": False, "bm25_off": True},
        "BM25 + HNSW (RRF)": {"colbert": False, "graph": False},
        "Full Pipeline": {"colbert": True, "graph": True},
    }

    for config_name, flags in configs.items():
        print(f"\n{'─' * 50}")
        print(f"  Configuration: {config_name}")
        print(f"{'─' * 50}")

        cfg = SystemConfig()
        cfg.enable_colbert_rerank = flags.get("colbert", False)

        reranker = colbert if cfg.enable_colbert_rerank else NoopReranker()
        gf = graph_features if flags.get("graph", False) else None

        retriever = HybridRetriever(
            bm25=bm25,
            vector=vector,
            graph_features=gf,
            reranker=reranker,
            profiles=profiles,
            config=cfg,
        )

        total_p, total_r, total_ndcg, total_mrr = 0.0, 0.0, 0.0, 0.0

        for query, relevant_ids in test_queries:
            t0 = time.time()
            results, analysis = retriever.search(query, top_k=k)
            latency = (time.time() - t0) * 1000

            retrieved_ids = [res.profile.id for res in results]

            p = precision_at_k(retrieved_ids, relevant_ids, k)
            r = recall_at_k(retrieved_ids, relevant_ids, k)
            n = ndcg_at_k(retrieved_ids, relevant_ids, k)
            m = mrr(retrieved_ids, relevant_ids)

            total_p += p
            total_r += r
            total_ndcg += n
            total_mrr += m

            print(f"  Q: '{query[:50]}...'")
            print(f"    P@{k}={p:.2f}  R@{k}={r:.2f}  nDCG@{k}={n:.2f}  MRR={m:.2f}  ({latency:.0f}ms)")

        n_q = len(test_queries)
        print(f"\n  AVG: P@{k}={total_p/n_q:.2f}  R@{k}={total_r/n_q:.2f}  nDCG@{k}={total_ndcg/n_q:.2f}  MRR={total_mrr/n_q:.2f}")

    print(f"\n{'=' * 60}")
    print("  Evaluation Complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    evaluate_system()
