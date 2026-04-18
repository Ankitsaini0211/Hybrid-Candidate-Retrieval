"""
Final Scoring Layer.

After ColBERT re-ranking, this module computes the true final score
as a weighted sum of ALL signals:

    final = w.bm25 * bm25 + w.faiss * faiss + w.graph * graph_feat + w.colbert * colbert

This replaces the old approach where RRF score was the final score.
"""
from typing import Dict, Any, List
from src.core.config import DynamicWeights


class FinalScorer:
    """Compute weighted hybrid score from all pipeline signals."""

    @staticmethod
    def score(
        candidate: Dict[str, Any],
        weights: DynamicWeights,
        analysis: Any = None,
    ) -> float:
        """
        Combine all component scores using dynamic weights.
        """
        scores = candidate.get("scores", {})
        graph_feat = candidate.get("graph_features", {})

        bm25_val = scores.get("bm25", 0.0)
        faiss_val = scores.get("faiss", 0.0)
        colbert_val = candidate.get("colbert_score", 0.0)

        # Graph signal: weighted combination of graph features
        graph_val = (
            graph_feat.get("skill_overlap", 0.0) * 0.4 +
            graph_feat.get("role_similarity", 0.0) * 0.3 +
            graph_feat.get("graph_distance", 0.0) * 0.2 +
            graph_feat.get("co_occurrence_score", 0.0) * 0.1
        )

        final = (
            weights.bm25 * bm25_val +
            weights.faiss * faiss_val +
            weights.graph * graph_val +
            weights.colbert * colbert_val
        )

        # Experience Penalty (Soft)
        if analysis and analysis.min_experience > 0:
            cand_exp = candidate["profile"].years_of_experience
            if cand_exp < analysis.min_experience:
                # Multiply by 0.1 to push under-qualified candidates to the bottom
                final *= 0.1

        return final

    @staticmethod
    def score_breakdown(
        candidate: Dict[str, Any],
        weights: DynamicWeights,
    ) -> Dict[str, float]:
        """Return a dict of each weighted component for explainability."""
        scores = candidate.get("scores", {})
        graph_feat = candidate.get("graph_features", {})

        bm25_val = scores.get("bm25", 0.0)
        faiss_val = scores.get("faiss", 0.0)
        colbert_val = candidate.get("colbert_score", 0.0)
        graph_val = (
            graph_feat.get("skill_overlap", 0.0) * 0.4 +
            graph_feat.get("role_similarity", 0.0) * 0.3 +
            graph_feat.get("graph_distance", 0.0) * 0.2 +
            graph_feat.get("co_occurrence_score", 0.0) * 0.1
        )

        return {
            "bm25_raw": bm25_val,
            "bm25_weighted": weights.bm25 * bm25_val,
            "faiss_raw": faiss_val,
            "faiss_weighted": weights.faiss * faiss_val,
            "graph_raw": graph_val,
            "graph_weighted": weights.graph * graph_val,
            "colbert_raw": colbert_val,
            "colbert_weighted": weights.colbert * colbert_val,
        }
