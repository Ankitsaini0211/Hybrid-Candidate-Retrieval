"""
Modular Feature Pipeline.

Decouples text construction from indexers.  Both BM25 and VectorIndexer
call ``FeaturePipeline.build_text(profile)`` instead of duplicating
text-assembly logic internally.

To add a new feature type, implement ``FeatureExtractor`` and register it.
"""
from typing import List, Protocol
from src.models.schema import CandidateProfile


# ── Protocol ───────────────────────────────────────────────────────────
class FeatureExtractor(Protocol):
    """Any class that can pull text from a CandidateProfile."""
    def extract(self, profile: CandidateProfile) -> str: ...


# ── Concrete Extractors ───────────────────────────────────────────────
class AttributeExtractor:
    """Extracts all dynamic attributes (universal schema)."""

    def extract(self, profile: CandidateProfile) -> str:
        parts: List[str] = []
        for attr in (profile.attributes or []):
            vals = " ".join(str(v) for v in attr.value)
            parts.append(f"{attr.key}: {vals}")
            parts.append(vals)          # repeat raw values for denser signal
        return " ".join(parts)


class SkillExtractor:
    """Extracts legacy skill fields (core + secondary)."""

    def extract(self, profile: CandidateProfile) -> str:
        parts: List[str] = []
        if profile.core_skills:
            parts.append(f"Skills: {profile.core_skills}")
        if profile.secondary_skills:
            parts.append(f"Secondary Skills: {profile.secondary_skills}")
        return " ".join(parts)


class RoleExtractor:
    """Extracts potential roles."""

    def extract(self, profile: CandidateProfile) -> str:
        if profile.potential_roles:
            return f"Roles: {profile.potential_roles}"
        return ""


class SummaryExtractor:
    """Extracts the rich natural-language skill summary."""

    def extract(self, profile: CandidateProfile) -> str:
        return profile.skill_summary or ""


class ExperienceExtractor:
    """Extracts years of experience as text."""

    def extract(self, profile: CandidateProfile) -> str:
        if profile.years_of_experience and profile.years_of_experience > 0:
            return f"Experience: {profile.years_of_experience} years"
        return ""


# ── Pipeline ───────────────────────────────────────────────────────────
class FeaturePipeline:
    """Compose multiple extractors into a single text builder."""

    def __init__(self, extractors: List[FeatureExtractor] | None = None):
        if extractors is None:
            # Default full pipeline
            extractors = [
                AttributeExtractor(),
                SkillExtractor(),
                RoleExtractor(),
                SummaryExtractor(),
                ExperienceExtractor(),
            ]
        self.extractors = extractors

    def build_text(self, profile: CandidateProfile) -> str:
        """Build a single text representation by running all extractors."""
        parts = [ext.extract(profile) for ext in self.extractors]
        return " ".join(p for p in parts if p).strip()

    def build_texts(self, profiles: List[CandidateProfile]) -> List[str]:
        """Batch version."""
        return [self.build_text(p) for p in profiles]
