"""
Hybrid Retriever — REWRITTEN for the new pipeline.

Flow:
  1. Query → QueryAnalyzer (intent + entities)
  2. WeightManager (intent → dynamic weights)
  3. Stage 1: BM25 + HNSW parallel retrieval
  4. Weighted RRF fusion → Top-K candidates (~100)
  5. Graph Feature Enrichment (Neo4j on top-K only)
  6. Stage 2: ColBERT re-ranking
  7. Final Hybrid Score (weighted sum of all signals)
  8. Explanation + Answer Assembly

Key change: Neo4j is NO LONGER a parallel retriever.
It is a feature augmenter that runs only on top-K candidates.
"""
from typing import List, Dict, Any, Optional

from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.vector_indexer import VectorIndexer
from src.core.query_analyzer import QueryAnalyzer, QueryAnalysis
from src.core.weight_manager import WeightManager
from src.core.graph_features import GraphFeatureExtractor
from src.core.reranker import Reranker, NoopReranker
from src.core.scoring import FinalScorer
from src.core.explanation_generator import ExplanationGenerator
from src.core.answer_assembler import AnswerAssembler
from src.core.config import SystemConfig, DynamicWeights
from src.models.schema import CandidateProfile, SearchResultItem


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Indexer,
        vector: VectorIndexer,
        graph_features: Optional[GraphFeatureExtractor],
        reranker: Optional[Reranker] = None,
        profiles: Optional[List[CandidateProfile]] = None,
        config: Optional[SystemConfig] = None,
    ):
        self.bm25 = bm25
        self.vector = vector
        self.graph_features = graph_features
        self.reranker = reranker or NoopReranker()
        self.config = config or SystemConfig()

        self.query_analyzer = QueryAnalyzer(profiles or [])
        self.weight_manager = WeightManager()
        self.explainer = ExplanationGenerator()
        self.answer_assembler = AnswerAssembler()

    # ------------------------------------------------------------------
    # Weighted Reciprocal Rank Fusion (Stage 1 only: BM25 + HNSW)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Weighted Reciprocal Rank Fusion (Stage 1 only: BM25 + HNSW)
    # ------------------------------------------------------------------
    def _normalize_scores(self, results: List[Dict]):
        """Min-Max normalization in-place to ensure comparable scores (0 to 1)."""
        if not results: return
        raw_scores = [r["score"] for r in results]
        min_s, max_s = min(raw_scores), max(raw_scores)
        
        for r in results:
            if max_s > min_s:
                r["score"] = (r["score"] - min_s) / (max_s - min_s + 1e-8)
            else:
                # If all scores are equal, assign 1.0 (they are all equally 'the best')
                r["score"] = 1.0

    def _weighted_rrf(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        weights: DynamicWeights,
    ) -> List[Dict[str, Any]]:
        """Fuse BM25 and HNSW results with weighted RRF."""
        k = self.config.rrf_k
        candidates: Dict[str, Dict[str, Any]] = {}

        def add_results(results: List[Dict], source: str, weight: float):
            for rank, res in enumerate(results):
                pid = res["profile"].id
                if pid not in candidates:
                    candidates[pid] = {
                        "profile": res["profile"],
                        "scores": {"bm25": 0.0, "faiss": 0.0}, # Initialize
                        "rrf_score": 0.0,
                    }
                candidates[pid]["scores"][source] = res["score"]
                candidates[pid]["rrf_score"] += weight * (1.0 / (k + rank + 1))

        add_results(bm25_results, "bm25", weights.bm25)
        add_results(vector_results, "faiss", weights.faiss)

        # Sort by weighted RRF score
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: c["rrf_score"],
            reverse=True,
        )
        return sorted_candidates

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------
    def search(
        self, query: str, top_k: int = 10, explain: bool = True
    ) -> List[SearchResultItem]:
        # ── 1. Query Analysis ──────────────────────────────────────────
        analysis = self.query_analyzer.analyze(query)

        # ── 2. Dynamic Weights ─────────────────────────────────────────
        if self.config.dynamic_weighting:
            weights = self.weight_manager.get_weights(analysis.intent)
        else:
            weights = self.config.default_weights

        # ── 3. Stage 1: Parallel retrieval (BM25 + HNSW only) ─────────
        fetch_k = 100  # Hard cap to reduce latency
        
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25.search, analysis.expanded_query, fetch_k)
            vector_future = executor.submit(self.vector.search, analysis.expanded_query, fetch_k)
            
            bm25_res = bm25_future.result()
            vector_res = vector_future.result()

        # Normalize scores BEFORE fusion
        self._normalize_scores(bm25_res)
        self._normalize_scores(vector_res)

        # ── 4. Weighted RRF Fusion → Initial Candidates ────────────────
        fused = self._weighted_rrf(bm25_res, vector_res, weights)

        # ── 5. HARD FILTER: Experience (CRITICAL FIX) ──────────────────
        applied_filters = []
        if analysis.min_experience > 0:
            filtered = [
                c for c in fused 
                if c["profile"].years_of_experience >= analysis.min_experience
            ]
            
            if not filtered and fused:
                # Fallback: pick the MOST experienced candidates anyway
                fused = sorted(fused, key=lambda x: x["profile"].years_of_experience, reverse=True)[:10]
                applied_filters.append(f"Experience ≥ {analysis.min_experience} (Soft Fallback)")
            else:
                fused = filtered
                applied_filters.append(f"Experience ≥ {analysis.min_experience} years (Enforced)")
        else:
            applied_filters.append("No Experience Constraint")

        # ── 6. Stage 2 Prep: Trim to top candidates for heavy lifting ──
        # Reduce rerank_k to 10 for < 1.5s latency on CPU
        rerank_k = min(10, len(fused))
        fused = fused[:rerank_k]

        if not fused:
            return [], analysis

        # ── 7. Graph Feature Enrichment (Neo4j on top-K only) ──────────
        if self.graph_features:
            candidate_ids = [c["profile"].id for c in fused]
            graph_feats = self.graph_features.batch_extract(analysis, candidate_ids)
            for c in fused:
                c["graph_features"] = graph_feats.get(c["profile"].id, {})
        else:
            for c in fused:
                c["graph_features"] = {}

        # ── 8. Stage 2: ColBERT Re-ranking ─────────────────────────────
        if self.config.enable_colbert_rerank:
            fused = self.reranker.rerank(analysis.expanded_query, fused, rerank_k)
            # Normalize ColBERT scores
            raw_colbert = [c.get("colbert_score", 0.0) for c in fused]
            if raw_colbert:
                min_c, max_c = min(raw_colbert), max(raw_colbert)
                for c in fused:
                    val = c.get("colbert_score", 0.0)
                    if max_c > min_c:
                        c["colbert_score"] = (val - min_c) / (max_c - min_c + 1e-8)
                    else:
                        c["colbert_score"] = 1.0

        # ── 9. Final Hybrid Score ──────────────────────────────────────
        for c in fused:
            c["final_score"] = FinalScorer.score(c, weights, analysis)
            c["score_breakdown"] = FinalScorer.score_breakdown(c, weights)

        # Sort by final score
        fused.sort(key=lambda c: c["final_score"], reverse=True)

        # ── 10. Smart Thresholding ─────────────────────────────────────
        top_score = fused[0]["final_score"] if fused else 0.0
        threshold = max(self.config.threshold_floor, top_score * self.config.threshold_ratio)

        # ── 11. Build results + Dedup ──────────────────────────────────
        final_results: List[SearchResultItem] = []
        seen_ids = set()

        for c in fused:
            if c["final_score"] < threshold:
                continue

            profile = c["profile"]
            
            # Dedup check
            if profile.id in seen_ids:
                continue
            seen_ids.add(profile.id)

            explanation_detail = None
            explanation_str = None

            if explain:
                explanation_detail = self.explainer.generate(
                    profile=profile,
                    query=query,
                    scores=c["scores"],
                    graph_features=c.get("graph_features", {}),
                    colbert_score=c.get("colbert_score"),
                    reranker_model=getattr(self.reranker, "model_name", None),
                    weights_used=weights,
                )
                explanation_str = explanation_detail.summary

            # Template answer
            answer = None
            if self.config.enable_template_answers:
                answer = self.answer_assembler.assemble(analysis, profile, c["final_score"])

            final_results.append(
                SearchResultItem(
                    profile=profile,
                    score=c["final_score"],
                    score_breakdown=c.get("score_breakdown"),
                    explanation=explanation_str,
                    explanation_detail=explanation_detail,
                    answer=answer,
                )
            )

        # Add filters to analysis for UI display
        analysis_dict = analysis.as_dict()
        analysis_dict["applied_filters"] = applied_filters

        return final_results, analysis_dict
