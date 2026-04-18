"""
Weight Manager — intent-adaptive dynamic weight selection.

Maps query intent → per-signal weights so that skill-focused queries
boost BM25, semantic queries boost FAISS, experience queries boost graph
features, etc.  Replaces equal-weight / hardcoded scoring.
"""
from src.core.config import DynamicWeights


# ── Weight presets per intent (Aggressive & Dynamic) ────────────────────
_WEIGHT_PRESETS = {
    "role_skill_experience_search": DynamicWeights(bm25=1.2, faiss=1.1, graph=1.5, colbert=1.2),
    "skill_experience_search":      DynamicWeights(bm25=1.5, faiss=0.8, graph=1.5, colbert=1.0),
    "role_and_skill_search":        DynamicWeights(bm25=1.2, faiss=1.2, graph=1.0, colbert=1.5),
    "skill_search":                 DynamicWeights(bm25=1.8, faiss=0.5, graph=0.5, colbert=1.5),
    "role_search":                  DynamicWeights(bm25=0.5, faiss=1.8, graph=0.8, colbert=1.5),
    "experience_query":             DynamicWeights(bm25=0.5, faiss=0.5, graph=2.0, colbert=0.8),
    "semantic_search":              DynamicWeights(bm25=0.3, faiss=2.0, graph=0.5, colbert=1.8),
    "general_search":               DynamicWeights(bm25=1.0, faiss=1.0, graph=1.0, colbert=1.0),
}


class WeightManager:
    """Select scoring weights based on detected query intent."""

    def __init__(self, presets: dict | None = None):
        self.presets = presets or _WEIGHT_PRESETS

    def get_weights(self, intent: str) -> DynamicWeights:
        """Return the weight profile for the given intent type."""
        return self.presets.get(intent, self.presets["general_search"])

    def register_preset(self, intent: str, weights: DynamicWeights):
        """Hot-add a new weight preset at runtime."""
        self.presets[intent] = weights
