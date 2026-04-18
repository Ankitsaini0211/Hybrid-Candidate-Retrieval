"""
Query Analyzer — replaces the old IntentDetector.

Richer analysis: extracts intent type, skills, roles, experience
requirements, and builds an expanded query for downstream retrieval.
"""
import re
from typing import List, Set, Optional
from src.models.schema import CandidateProfile


class QueryAnalysis:
    """Structured output of query analysis."""
    __slots__ = (
        "original_query", "intent", "extracted_skills", "extracted_roles",
        "min_experience", "expanded_query",
    )

    def __init__(
        self,
        original_query: str,
        intent: str,
        extracted_skills: List[str],
        extracted_roles: List[str],
        min_experience: Optional[float],
        expanded_query: str,
    ):
        self.original_query = original_query
        self.intent = intent
        self.extracted_skills = extracted_skills
        self.extracted_roles = extracted_roles
        self.min_experience = min_experience
        self.expanded_query = expanded_query

    def as_dict(self):
        return {
            "original_query": self.original_query,
            "intent_type": self.intent,
            "extracted_skills": self.extracted_skills,
            "extracted_roles": self.extracted_roles,
            "min_experience": self.min_experience,
            "expanded_query": self.expanded_query,
        }


# ── Experience extraction patterns ────────────────────────────────────
_EXP_PATTERNS = [
    re.compile(r"(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?", re.I),
    re.compile(r"(?:at\s+least|minimum|min)\s+(\d+)\s*(?:years?|yrs?)", re.I),
    re.compile(r"(?:experience|exp)\s*[>:=]+\s*(\d+)", re.I),
    re.compile(r"more than\s*(\d+)", re.I),
]


class QueryAnalyzer:
    """
    Data-driven query analyzer with structured extraction.
    """

    def __init__(self, profiles: List[CandidateProfile]):
        self.skill_vocab: Set[str] = set()
        self.role_vocab: Set[str] = set()
        self._build_vocabulary(profiles)
        
        # Skill-synonym map for better extraction
        self.skill_synonyms = {
            "python": ["python", "django", "flask", "fastapi"],
            "java": ["java", "spring", "spring boot"],
            "ml": ["machine learning", "deep learning", "ai", "pytorch", "tensorflow"],
            "javascript": ["js", "react", "node", "angular", "vue"],
            "sql": ["sql", "mysql", "postgresql", "oracle", "database"],
            "aws": ["aws", "cloud", "ec2", "s3", "lambda"],
        }

    def _build_vocabulary(self, profiles: List[CandidateProfile]):
        for p in profiles:
            for col in [p.core_skills, p.secondary_skills]:
                for term in self._extract_terms(col or ""):
                    self.skill_vocab.add(term)
            for term in self._extract_terms(p.potential_roles or ""):
                self.role_vocab.add(term)

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        if not text: return []
        return [t.split("(")[0].strip().lower() for t in text.split(",") if len(t.strip()) > 2]

    def _extract_experience(self, query: str) -> float:
        for pat in _EXP_PATTERNS:
            m = pat.search(query)
            if m:
                return float(m.group(1))
        return 0.0

    def _clean_expand(self, skills: List[str], roles: List[str]) -> str:
        """Controlled expansion: only expand skills and roles, ignore noise tokens."""
        expanded = set(skills + roles)
        for s in skills:
            if s in self.skill_synonyms:
                expanded.update(self.skill_synonyms[s])
        return " ".join(list(expanded))

    def analyze(self, query: str) -> QueryAnalysis:
        query_lower = query.lower()
        
        # 1. Extract Skills (Vocabulary + Synonym matches)
        matched_skills = set()
        for skill in self.skill_vocab:
            if skill in query_lower:
                matched_skills.add(skill)
        for canonical, syns in self.skill_synonyms.items():
            for s in syns:
                if s in query_lower:
                    matched_skills.add(canonical)
                    break
        
        # 2. Extract Roles
        matched_roles = set()
        for role in self.role_vocab:
            if role in query_lower:
                matched_roles.add(role)
        
        # 3. Extract Experience
        min_exp = self._extract_experience(query)
        
        # 4. Refined Intent Detection
        skills_list = list(matched_skills)
        roles_list = list(matched_roles)
        
        if roles_list and skills_list and min_exp > 0:
            intent = "role_skill_experience_search"
        elif skills_list and min_exp > 0:
            intent = "skill_experience_search"
        elif roles_list and skills_list:
            intent = "role_and_skill_search"
        elif roles_list:
            intent = "role_search"
        elif skills_list:
            intent = "skill_search"
        elif min_exp > 0:
            intent = "experience_query"
        else:
            intent = "general_search"

        # 5. Clean Expansion (ignore 'more than', 'years', etc.)
        expanded = self._clean_expand(skills_list, roles_list)
        if not expanded:
            # Fallback for general search
            query_clean = re.sub(r"[^a-z0-9\s]", " ", query_lower)
            expanded = " ".join([t for t in query_clean.split() if len(t) > 2])

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            extracted_skills=skills_list[:5],
            extracted_roles=roles_list[:5],
            min_experience=min_exp,
            expanded_query=expanded,
        )
