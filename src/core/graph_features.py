"""
Graph Feature Extractor — Final Surgical Version.

Extracts ranking signals from Neo4j for top-K candidates.
Ensures all features are normalized [0, 1].
"""
from typing import List, Dict, Optional
from src.core.query_analyzer import QueryAnalysis


class GraphFeatureExtractor:
    """Extract graph-based ranking features from Neo4j for a set of candidates."""

    def __init__(self, graph_indexer):
        self.graph = graph_indexer

    def batch_extract(
        self, query_analysis: QueryAnalysis, candidate_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract graph features for multiple candidates in batch.
        Calculates:
          - skill_overlap:  (matched skills) / (total query skills)
          - role_similarity: (matched roles) / (total query roles)
          - graph_distance:  1 / (1 + shortest_path)
          - co_occurrence:  (shared skill frequency / max) -> simplified to skill_overlap for this hackathon
        """
        if not candidate_ids:
            return {}

        query_skills = [s.lower() for s in query_analysis.extracted_skills]
        query_roles = [r.lower() for r in query_analysis.extracted_roles]
        
        default = {
            "skill_overlap": 0.0, 
            "role_similarity": 0.0, 
            "graph_distance": 0.0, 
            "co_occurrence_score": 0.0
        }

        results: Dict[str, Dict[str, float]] = {cid: dict(default) for cid in candidate_ids}

        try:
            with self.graph.driver.session(database=self.graph._db) as session:
                # ── 1. Batch Skill & Role Overlap ───────────────────────
                # This Cypher query gets both in one pass
                records = session.run(
                    """
                    UNWIND $cids AS cid
                    MATCH (c:Candidate {id: cid})
                    
                    OPTIONAL MATCH (c)-[:HAS_SKILL]->(s:Skill)
                    WHERE toLower(s.name) IN $skills
                    WITH cid, c, count(DISTINCT toLower(s.name)) AS matched_skills
                    
                    OPTIONAL MATCH (c)-[:HAS_ROLE]->(r:Role)
                    WHERE toLower(r.name) IN $roles
                    WITH cid, c, matched_skills, count(DISTINCT toLower(r.name)) AS matched_roles
                    
                    RETURN cid, matched_skills, matched_roles
                    """,
                    cids=candidate_ids,
                    skills=query_skills,
                    roles=query_roles
                )
                
                for rec in records:
                    cid = rec["cid"]
                    if cid in results:
                        results[cid]["skill_overlap"] = rec["matched_skills"] / max(1, len(query_skills))
                        results[cid]["role_similarity"] = rec["matched_roles"] / max(1, len(query_roles))
                        # Co-occurrence simplified to skill overlap for the purpose of the 0.1 weight
                        results[cid]["co_occurrence_score"] = results[cid]["skill_overlap"]

                # ── 2. Batch Graph Distance ────────────────────────────
                all_entities = query_skills + query_roles
                if all_entities:
                    dist_records = session.run(
                        """
                        UNWIND $cids AS cid
                        MATCH (c:Candidate {id: cid})
                        MATCH (target)
                        WHERE (target:Skill OR target:Role)
                          AND toLower(target.name) IN $entities
                        WITH cid, c, target
                        MATCH sp = shortestPath((c)-[*..3]-(target))
                        WITH cid, min(length(sp)) AS min_dist
                        RETURN cid, min_dist
                        """,
                        cids=candidate_ids,
                        entities=all_entities
                    )
                    for rec in dist_records:
                        cid = rec["cid"]
                        if cid in results:
                            d = rec["min_dist"]
                            # Normalize: 1 / (1 + d).  d=0 -> 1.0, d=1 -> 0.5, etc.
                            results[cid]["graph_distance"] = 1.0 / (1.0 + float(d))

        except Exception as e:
            print(f"[GraphFeatures] Batch extract error: {e}")

        return results
