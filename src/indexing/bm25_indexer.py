from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any
from src.models.schema import CandidateProfile

STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "in", "to", "for",
    "with", "is", "it", "at", "on", "by", "as", "be", "this",
    "manager", "specialist", "coordinator", "associate", "senior",
    "junior", "lead", "expert", "professional"
}

SYNONYMS = {
    "frontend":   "frontend",
    "front-end":  "frontend",
    "back-end":   "backend",
    "fullstack":  "fullstack",
    "full-stack": "fullstack",
    "html5":      "html",
    "css3":       "css",
    "js":         "javascript",
    "react.js":   "react",
    "node.js":    "nodejs",
    "vue.js":     "vue",
    "postgres":   "postgresql",
    "ml":         "machine learning",
    "ai":         "artificial intelligence",
}

class BM25Indexer:
    def __init__(self):
        self.bm25_corpus = []
        self.corpus: list = []
        self.profiles = []
        self.bm25 = None
        self._pipeline = None  # Optional FeaturePipeline

    def set_pipeline(self, pipeline):
        """Attach a shared FeaturePipeline for text construction."""
        self._pipeline = pipeline

    def _preprocess(self, text: str) -> List[str]:
        if not text:
            return []

        # 1. Lowercase
        text = text.lower()

        # 2. Normalize special tech tokens before stripping punctuation
        text = text.replace("c++", "cplusplus")
        text = text.replace("c#",  "csharp")
        text = text.replace(".net", "dotnet")
        text = text.replace(".js",  "js")

        # 3. Remove all punctuation — keeps alphanumeric and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # 4. Tokenize on whitespace
        tokens = text.split()

        # 5. Remove stopwords and single-char tokens
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

        # 6. Synonym normalization
        tokens = [SYNONYMS.get(t, t) for t in tokens]

        return tokens

    def _build_text(self, profile) -> str:
        """Build text for a profile using pipeline if available, else legacy."""
        if self._pipeline:
            return self._pipeline.build_text(profile)
        text_parts = []
        for a in (profile.attributes or []):
            vals = " ".join(str(v) for v in a.value)
            text_parts.append(f"{a.key}: {vals}")
        if profile.skill_summary:
            text_parts.append(profile.skill_summary)
        return " ".join(text_parts)

    def index(self, profiles: List[CandidateProfile]):
        self.profiles = profiles
        tokenized_corpus = []
        for p in profiles:
            text = self._build_text(p)
            tokenized_corpus.append(self._preprocess(text))
        
        self.corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_single(self, profile: CandidateProfile):
        """Add one profile incrementally and rebuild BM25."""
        self.profiles.append(profile)
        text = self._build_text(profile)
        self.corpus.append(self._preprocess(text))
        self.bm25 = BM25Okapi(self.corpus)

    def remove(self, profile_id: str) -> bool:
        """Remove a profile by ID and rebuild BM25."""
        idx = None
        for i, p in enumerate(self.profiles):
            if p.id == profile_id:
                idx = i
                break
        if idx is None:
            return False
        self.profiles.pop(idx)
        self.corpus.pop(idx)
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None
        return True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        
        tokenized_query = self._preprocess(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        query_set = set(tokenized_query)

        results = []
        for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
            # Hard filter: document MUST share at least one token with query
            doc_tokens = set(self.corpus[idx])
            if not (query_set & doc_tokens):
                continue  # Zero lexical overlap — skip completely

            results.append({
                "profile": self.profiles[idx],
                "score": float(scores[idx]),
                "technique": "bm25"
            })

            if len(results) >= top_k:
                break

        return results

