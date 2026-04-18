"""
Advanced Evaluation Script for Hybrid Retrieval System.

Calculates research-grade IR metrics:
- MRR (Mean Reciprocal Rank)
- Precision@5
- nDCG@10 (Normalized Discounted Cumulative Gain)
"""
import os
import sys
import math
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import SystemConfig
from src.core.data_loader import load_and_clean_data
from src.core.embedding_service import EmbeddingService
from src.core.feature_pipeline import FeaturePipeline
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.vector_indexer import VectorIndexer
from src.indexing.graph_indexer import GraphIndexer
from src.core.graph_features import GraphFeatureExtractor
from src.core.reranker import NoopReranker # Noop for faster eval
from src.core.retriever import HybridRetriever

def calculate_ndcg(found_rank, k=10):
    """Simple nDCG calculation for a single relevant document."""
    if found_rank == 0 or found_rank > k:
        return 0.0
    # DCG = rel / log2(rank + 1). rel = 1 for our ground truth.
    dcg = 1.0 / math.log2(found_rank + 1)
    # IDCG = 1.0 / log2(1 + 1) = 1.0 (since there is only 1 ground truth)
    return dcg

def run_eval():
    print(f"\n{'='*60}")
    print("  HYBRID RETRIEVAL SYSTEM — EVALUATION SUITE")
    print(f"{'='*60}\n")
    
    # 1. Setup
    config = SystemConfig()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "profiles.csv")
    profiles = load_and_clean_data(data_path)
    
    embedding_service = EmbeddingService(model_name=config.embedding_model)
    feature_pipeline = FeaturePipeline()
    
    print("Indexing profiles...")
    bm25 = BM25Indexer()
    bm25.set_pipeline(feature_pipeline)
    bm25.index(profiles)
    
    vector = VectorIndexer(embedding_service, feature_pipeline, config)
    vector.index_all(profiles)
    
    graph_indexer = GraphIndexer()
    try:
        graph_indexer.index(profiles, embedding_service=embedding_service)
        graph_features = GraphFeatureExtractor(graph_indexer)
    except:
        graph_features = None

    retriever = HybridRetriever(
        bm25=bm25,
        vector=vector,
        graph_features=graph_features,
        reranker=NoopReranker(), 
        profiles=profiles,
        config=config
    )

    # 2. Benchmark Queries & Ground Truth (10 Queries)
    benchmarks = [
        {"query": "python full stack developer", "truth": "234922"},
        {"query": "machine learning python", "truth": "234957"},
        {"query": "QA engineer selenium", "truth": "235314"},
        {"query": "talent acquisition specialist", "truth": "235138"},
        {"query": "SAP S/4HANA procurement", "truth": "181413"},
        {"query": "regulatory affairs manager cosmetics", "truth": "180869"},
        {"query": "technical lead microservices azure", "truth": "235205"},
        {"query": "backend engineer node.js express", "truth": "234950"},
        {"query": "manual tester defect management", "truth": "235314"},
        {"query": "software engineer python sql", "truth": "181457"}
    ]

    mrr_sum = 0
    p5_sum = 0
    ndcg_sum = 0
    
    print(f"Running benchmarks on {len(benchmarks)} queries...")
    
    for b in benchmarks:
        query = b["query"]
        truth_id = str(b["truth"])
        
        # We run with explain=False for speed in eval
        results, _ = retriever.search(query, top_k=10, explain=False)
        
        rank = 0
        found = False
        for i, res in enumerate(results):
            if str(res.profile.id) == truth_id:
                rank = i + 1
                found = True
                break
        
        if found:
            mrr_sum += 1.0 / rank
            if rank <= 5:
                p5_sum += 1
            ndcg_sum += calculate_ndcg(rank)
            print(f"  [PASS] '{query}'".ljust(45) + f"-> Rank {rank}")
        else:
            print(f"  [FAIL] '{query}'".ljust(45) + "-> Not in Top 10")

    avg_mrr = mrr_sum / len(benchmarks)
    avg_p5 = p5_sum / len(benchmarks)
    avg_ndcg = ndcg_sum / len(benchmarks)

    print(f"\n{'='*40}")
    print(f"  FINAL PERFORMANCE RESULTS")
    print(f"{'='*40}")
    print(f"  Mean Reciprocal Rank (MRR):   {avg_mrr:.3f}")
    print(f"  Precision@5:                  {avg_p5:.3f}")
    print(f"  nDCG@10:                      {avg_ndcg:.3f}")
    print(f"{'='*40}\n")
    print("Interpretation:")
    print(f"  MRR {avg_mrr:.2f} -> Average rank is {1/avg_mrr:.1f}")
    print(f"  P@5 {avg_p5:.2f} -> {avg_p5*100:.0f}% of top queries hit ground truth in top 5")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    run_eval()
