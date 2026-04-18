import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
from src.models.schema import CandidateProfile

class FAISSIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index_db = faiss.IndexFlatL2(self.dimension)
        self.profiles = []

    def index(self, profiles: List[CandidateProfile]):
        self.profiles = profiles
        texts = []
        for p in profiles:
            text_parts = []

            # Dynamic attributes (universal — works with any CSV schema)
            for a in (p.attributes or []):
                vals = " ".join(str(v) for v in a.value)
                text_parts.append(f"{a.key}: {vals}")
                text_parts.append(vals)  # boost: repeat for denser semantic embedding

            # Legacy high-signal fields for extra BERT weight
            if p.core_skills:
                text_parts.append(f"Skills: {p.core_skills}")
            if p.potential_roles:
                text_parts.append(f"Roles: {p.potential_roles}")
            if p.skill_summary:
                text_parts.append(p.skill_summary)

            texts.append(" ".join(text_parts))
        
        if not texts:
            return

        print(f"Embedding {len(texts)} documents for FAISS index...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        faiss.normalize_L2(embeddings) # Cosine similarity via L2 normalized
        self.index_db.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index_db.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index_db.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # valid index
                # Convert distance to similarity score
                similarity = 1.0 / (1.0 + distances[0][i])
                results.append({
                    "profile": self.profiles[idx],
                    "score": float(similarity),
                    "technique": "faiss"
                })
        return results
