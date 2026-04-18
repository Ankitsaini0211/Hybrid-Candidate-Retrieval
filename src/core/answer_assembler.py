"""
Answer Assembler — intent-driven template-based NLG.

Generates a human-readable answer sentence for each result
WITHOUT any LLM calls.  Uses query intent to select the right
template and fills it from profile fields.
"""
from typing import Optional
from src.models.schema import CandidateProfile
from src.core.query_analyzer import QueryAnalysis


class AnswerAssembler:
    """Template-based answer generation driven by query intent."""

    def assemble(
        self,
        query_analysis: QueryAnalysis,
        profile: CandidateProfile,
        score: float = 0.0,
    ) -> str:
        """Generate a contextual answer sentence for a search result."""

        name = profile.name or f"Candidate {profile.id}"
        intent = query_analysis.intent

        # ── Experience query ───────────────────────────────────────────
        if intent == "experience_query":
            exp = profile.years_of_experience or 0
            skills_str = self._get_top_skills(profile, 3)
            if skills_str:
                return (
                    f"{name} has {exp:.0f} years of experience, "
                    f"with expertise in {skills_str}."
                )
            return f"{name} has {exp:.0f} years of professional experience."

        # ── Skill search ───────────────────────────────────────────────
        if intent == "skill_search":
            matched = self._match_skills(query_analysis.extracted_skills, profile)
            if matched:
                return (
                    f"{name} is skilled in {', '.join(matched[:4])}"
                    f"{self._exp_suffix(profile)}."
                )
            # Fallback: show top skills
            skills_str = self._get_top_skills(profile, 3)
            return f"{name} has skills in {skills_str}{self._exp_suffix(profile)}."

        # ── Role search ────────────────────────────────────────────────
        if intent == "role_search":
            roles = self._get_roles(profile)
            if roles:
                return (
                    f"{name} is a strong match for {', '.join(roles[:3])}"
                    f"{self._exp_suffix(profile)}."
                )
            return f"{name} is a potential match for the queried role{self._exp_suffix(profile)}."

        # ── Role + Skill search ────────────────────────────────────────
        if intent == "role_and_skill_search":
            roles = self._get_roles(profile)
            matched = self._match_skills(query_analysis.extracted_skills, profile)
            role_str = f"as a {roles[0]}" if roles else ""
            skill_str = f"with skills in {', '.join(matched[:3])}" if matched else ""
            return f"{name} fits {role_str} {skill_str}{self._exp_suffix(profile)}.".strip()

        # ── General / fallback ─────────────────────────────────────────
        summary = (profile.skill_summary or "")[:150].strip()
        if summary:
            if not summary.endswith("."):
                summary += "..."
            return f"{name} — {summary}"
        return f"{name}{self._exp_suffix(profile)}."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_top_skills(profile: CandidateProfile, n: int = 3) -> str:
        skills = []
        if profile.core_skills:
            for s in profile.core_skills.split(","):
                clean = s.split("(")[0].strip()
                if clean and clean not in skills:
                    skills.append(clean)
        return ", ".join(skills[:n])

    @staticmethod
    def _get_roles(profile: CandidateProfile):
        if not profile.potential_roles:
            return []
        return [r.strip() for r in profile.potential_roles.split(",") if r.strip()][:3]

    @staticmethod
    def _match_skills(query_skills, profile: CandidateProfile):
        if not query_skills:
            return []
        profile_skills = set()
        for col in [profile.core_skills, profile.secondary_skills]:
            if col:
                for s in col.split(","):
                    profile_skills.add(s.split("(")[0].strip().lower())
        return [qs for qs in query_skills if qs.lower() in profile_skills]

    @staticmethod
    def _exp_suffix(profile: CandidateProfile) -> str:
        if profile.years_of_experience and profile.years_of_experience > 0:
            return f" with {profile.years_of_experience:.0f} years of experience"
        return ""
