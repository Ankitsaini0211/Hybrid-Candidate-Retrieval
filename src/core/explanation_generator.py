"""
Explanation Generator — updated for the new pipeline.

Now includes:
  - Lexical (BM25) matches
  - Semantic (FAISS) similarity
  - Graph feature signals (skill_overlap, role_sim, distance)
  - ColBERT re-ranker score
  - Dynamic weights used
"""
from typing import Dict, Any, List, Optional
from src.models.schema import CandidateProfile, ExplanationDetail, MethodType, GraphStep
from src.core.config import DynamicWeights


class ExplanationGenerator:
    def generate(
        self,
        profile: CandidateProfile,
        query: str,
        scores: Dict[str, Any],
        graph_features: Optional[Dict[str, float]] = None,
        colbert_score: Optional[float] = None,
        reranker_model: Optional[str] = None,
        weights_used: Optional[DynamicWeights] = None,
    ) -> ExplanationDetail:
        """
        Produces a structured ExplanationDetail covering every pipeline stage.
        """
        query_lower = query.lower()
        methods: List[MethodType] = []
        lexical_matches: List[str] = []
        graph_steps: List[GraphStep] = []

        bm25_score = scores.get("bm25", None)
        faiss_score = scores.get("faiss", None)
        graph_feat = graph_features or {}

        # ── Lexical (BM25) ─────────────────────────────────────────────
        if bm25_score is not None and bm25_score > 0:
            methods.append(MethodType.lexical)
            query_tokens = set(t for t in query_lower.split() if len(t) > 2)
            for attr in (profile.attributes or []):
                for item in attr.value:
                    clean = str(item).strip().lower()
                    if clean and any(t in clean or clean in t for t in query_tokens):
                        if clean not in lexical_matches:
                            lexical_matches.append(clean)
            # Fallback: legacy fields
            if not lexical_matches:
                for col in [profile.core_skills, profile.secondary_skills]:
                    if not col:
                        continue
                    for skill in col.split(","):
                        clean = skill.split("(")[0].strip().lower()
                        if clean and any(t in clean or clean in t for t in query_tokens):
                            if clean not in lexical_matches:
                                lexical_matches.append(clean)

        # ── Semantic (FAISS) ───────────────────────────────────────────
        semantic_similarity = None
        if faiss_score is not None and faiss_score > 0:
            methods.append(MethodType.semantic)
            semantic_similarity = float(faiss_score)

        # ── Graph Features ─────────────────────────────────────────────
        graph_score_val = None
        if graph_feat and any(v > 0 for v in graph_feat.values()):
            methods.append(MethodType.graph)
            graph_score_val = sum(graph_feat.values()) / max(1, len(graph_feat))

        # ── Re-ranker (ColBERT) ────────────────────────────────────────
        reranker_score_val = None
        if colbert_score is not None and colbert_score > 0:
            methods.append(MethodType.reranker)
            reranker_score_val = colbert_score

        # ── Human-readable summary ─────────────────────────────────────
        main_skill = lexical_matches[0] if lexical_matches else "relevant experience"
        summary = f"{profile.name} is skilled in {main_skill}."
        
        return ExplanationDetail(
            methods=methods,
            bm25_score=round(bm25_score, 4) if bm25_score is not None else None,
            faiss_score=round(faiss_score, 4) if faiss_score is not None else None,
            graph_score=round(graph_score_val, 4) if graph_score_val is not None else None,
            lexical_matches=lexical_matches[:6],
            semantic_similarity=semantic_similarity,
            graph_traversal=graph_steps,
            graph_features=graph_feat if graph_feat else None,
            reranker_score=round(reranker_score_val, 4) if reranker_score_val is not None else None,
            reranker_model=reranker_model,
            weights_used=weights_used.as_dict() if weights_used else None,
            summary=summary,
        )
