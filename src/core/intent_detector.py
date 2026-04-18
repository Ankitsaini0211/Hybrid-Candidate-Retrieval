"""
Data-driven IntentDetector.

Instead of hardcoded skill/role lists, vocabulary is extracted at runtime
from the actual candidate profiles loaded from CSV.
"""
from typing import Dict, Any, List, Set
from src.models.schema import CandidateProfile


class IntentDetector:
    def __init__(self, profiles: List[CandidateProfile]):
        self.skill_vocab: Set[str] = set()
        self.role_vocab: Set[str] = set()
        self._build_vocabulary(profiles)

    # ------------------------------------------------------------------
    # Vocabulary building
    # ------------------------------------------------------------------
    def _extract_terms(self, text: str) -> List[str]:
        if not text:
            return []
        terms = []
        for part in text.split(","):
            clean = part.split("(")[0].strip().lower()
            if clean and len(clean) > 2:
                terms.append(clean)
        return terms

    def _build_vocabulary(self, profiles: List[CandidateProfile]):
        for p in profiles:
            # Skills from core and secondary columns
            for col in [p.core_skills, p.secondary_skills]:
                for term in self._extract_terms(col or ""):
                    self.skill_vocab.add(term)
            # Roles from potential_roles column
            for term in self._extract_terms(p.potential_roles or ""):
                self.role_vocab.add(term)

        print(
            f"[IntentDetector] Vocabulary built from data: "
            f"{len(self.skill_vocab)} skills, {len(self.role_vocab)} roles."
        )

    # ------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------
    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Match query tokens against the data-driven skill and role vocabularies.
        A token matches a vocabulary term if either contains the other
        (handles partial matches like 'python' matching 'python programming').
        """
        query_lower = query.lower()
        query_tokens = [t.strip() for t in query_lower.split() if len(t.strip()) > 2]

        matched_skills: List[str] = []
        matched_roles: List[str] = []

        for token in query_tokens:
            # Check skills
            for skill in self.skill_vocab:
                if token in skill or skill in token:
                    if skill not in matched_skills:
                        matched_skills.append(skill)
            # Check roles
            for role in self.role_vocab:
                if token in role or role in token:
                    if role not in matched_roles:
                        matched_roles.append(role)

        # Also do a full-phrase scan for multi-word terms
        for skill in self.skill_vocab:
            if len(skill) > 4 and skill in query_lower and skill not in matched_skills:
                matched_skills.append(skill)
        for role in self.role_vocab:
            if len(role) > 4 and role in query_lower and role not in matched_roles:
                matched_roles.append(role)

        # Limit to top 5 most relevant matches (shorter = more specific)
        matched_skills = sorted(matched_skills, key=len)[:5]
        matched_roles = sorted(matched_roles, key=len)[:5]

        intent_type = "general_search"
        if matched_roles and not matched_skills:
            intent_type = "role_search"
        elif matched_skills and not matched_roles:
            intent_type = "skill_search"
        elif matched_roles and matched_skills:
            intent_type = "role_and_skill_search"

        extra = " ".join(matched_roles + matched_skills)
        return {
            "original_query": query,
            "intent_type": intent_type,
            "extracted_roles": matched_roles,
            "extracted_skills": matched_skills,
            "expanded_query": f"{query} {extra}".strip(),
        }
