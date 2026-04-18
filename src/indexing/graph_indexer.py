"""
Neo4j-backed Knowledge Graph Indexer.

Graph schema:
  (:Candidate)-[:HAS_SKILL]->(:Skill)
  (:Candidate)-[:HAS_ROLE]->(:Role)

This enables 2-hop traversal paths like:
  Query → [mongodb] → [Backend Developer] → Candidate

Connection is read from environment variables (Aura-compatible):
  NEO4J_URI       — e.g. neo4j+s://xxxx.databases.neo4j.io
  NEO4J_USERNAME  — usually 'neo4j'
  NEO4J_PASSWORD  — your Aura or local password
  NEO4J_DATABASE  — defaults to 'neo4j'
"""
import os
import re as _re
from typing import List, Dict, Any
from neo4j import GraphDatabase
from src.models.schema import CandidateProfile

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Works fine without it when env vars are set directly

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class GraphIndexer:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._db = NEO4J_DATABASE
        self.profiles: List[CandidateProfile] = []
        self._profile_map: Dict[str, CandidateProfile] = {}
        
        # State for semantic node matching
        self.node_names: List[str] = []
        self.semantic_model = None
        self.node_embeddings = None

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """Split 'Python (Advanced), Java (Beginner)' → ['python', 'java']"""
        if not text:
            return []
        result = []
        for part in text.split(","):
            clean = part.split("(")[0].strip().lower()
            if clean:
                result.append(clean)
        return result

    # ------------------------------------------------------------------
    # Indexing: Candidate → Skill, Candidate → Role  (BATCHED)
    # ------------------------------------------------------------------
    def index(self, profiles: List[CandidateProfile], embedding_service=None):
        self.profiles = profiles
        self._profile_map = {p.id: p for p in profiles}

        all_unique_nodes: set = set()

        # ── Build batch payloads ───────────────────────────────────────
        candidate_rows = []
        skill_edges = []    # [{cid, skill_name}]
        role_edges  = []    # [{cid, role_name}]

        for p in profiles:
            candidate_rows.append({
                "id": p.id,
                "name": p.name or "",
                "canonical_id": p.canonical_id or "",
                "entity_type": p.entity_type or "candidate",
            })

            # skills from core + secondary
            for col in [p.core_skills, p.secondary_skills]:
                for term in self._extract_terms(col or ""):
                    if term:
                        skill_edges.append({"cid": p.id, "skill": term})
                        all_unique_nodes.add(term)

            # roles
            for term in self._extract_terms(p.potential_roles or ""):
                if term:
                    role_edges.append({"cid": p.id, "role": term})
                    all_unique_nodes.add(term)

        with self.driver.session(database=self._db) as session:
            # Constraints (idempotent)
            for constraint in [
                "CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT skill_name   IF NOT EXISTS FOR (s:Skill)     REQUIRE s.name IS UNIQUE",
                "CREATE CONSTRAINT role_name    IF NOT EXISTS FOR (r:Role)      REQUIRE r.name IS UNIQUE",
            ]:
                try:
                    session.run(constraint)
                except Exception:
                    pass  # constraint may already exist

            # 1 query — upsert all candidates
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Candidate {id: row.id})
                SET c.name = row.name,
                    c.canonical_id = row.canonical_id,
                    c.entity_type = row.entity_type
                """,
                rows=candidate_rows,
            )
            print(f"[GraphIndexer] Upserted {len(candidate_rows)} candidates.")

            # 1 query — upsert all skills + edges
            if skill_edges:
                session.run(
                    """
                    UNWIND $edges AS edge
                    MERGE (s:Skill {name: edge.skill})
                    WITH s, edge
                    MATCH (c:Candidate {id: edge.cid})
                    MERGE (c)-[:HAS_SKILL]->(s)
                    """,
                    edges=skill_edges,
                )
                print(f"[GraphIndexer] Upserted {len(skill_edges)} skill edges.")

            # 1 query — upsert all roles + edges
            if role_edges:
                session.run(
                    """
                    UNWIND $edges AS edge
                    MERGE (r:Role {name: edge.role})
                    WITH r, edge
                    MATCH (c:Candidate {id: edge.cid})
                    MERGE (c)-[:HAS_ROLE]->(r)
                    """,
                    edges=role_edges,
                )
                print(f"[GraphIndexer] Upserted {len(role_edges)} role edges.")

        # ── Semantic node embeddings (reuse shared service if provided) ──
        self.node_names = list(all_unique_nodes)
        if self.node_names:
            try:
                import torch
                if embedding_service is not None:
                    # Reuse the already-loaded EmbeddingService — no second model load
                    self.node_embeddings = torch.tensor(
                        embedding_service.encode(self.node_names, normalize=False)
                    )
                    self.semantic_model = embedding_service.model  # raw SentenceTransformer
                else:
                    from sentence_transformers import SentenceTransformer
                    self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                    self.node_embeddings = self.semantic_model.encode(
                        self.node_names, convert_to_tensor=True
                    )
                print(f"[GraphIndexer] Embedded {len(self.node_names)} graph nodes.")
            except Exception as e:
                print(f"[GraphIndexer] Warning: node embeddings failed ({e})")

        print(f"[GraphIndexer] Done — {len(profiles)} candidates indexed into Neo4j.")

    # ------------------------------------------------------------------
    # Search: Agnistic Multi-Hop Traversal 
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Phase 1 — direct node match:
          Query tokens → matched dynamic nodes → connected Candidates

        Phase 2 — indirect multi-hop generic traversal:
          Find candidates connected to ANY intermediate shared graph node
        """
        tokens = [t.strip() for t in query.lower().split() if len(t.strip()) > 2]
        seen: set = set()
        unique_tokens = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)
        tokens = unique_tokens

        if not tokens:
            return []

        escaped = [_re.escape(t) for t in tokens]
        regex_parts = escaped.copy()
        semantic_proxies = {}

        if self.semantic_model is not None and self.node_embeddings is not None:
            import torch
            from sentence_transformers import util
            for t in tokens:
                t_lower = t.lower()
                exact_match = any(t_lower in n.lower() for n in self.node_names)
                if exact_match:
                    continue
                
                with torch.no_grad():
                    t_emb = self.semantic_model.encode(t, convert_to_tensor=True)
                    sims = util.cos_sim(t_emb, self.node_embeddings)[0]
                    best_idx = torch.argmax(sims).item()
                    best_score = sims[best_idx].item()
                    
                    if best_score > 0.4:
                        best_node = self.node_names[best_idx]
                        regex_parts.append(_re.escape(best_node))
                        semantic_proxies[best_node] = t

        regex = "(?i)(" + "|".join(regex_parts) + ")"

        with self.driver.session(database=self._db) as session:
            # ── Phase 1: Relation-Agnostic Direct Links ────────
            direct_results = session.run(
                """
                MATCH (s)
                WHERE s.name =~ $regex OR s.id =~ $regex
                MATCH (c:Candidate)-[]->(s)
                RETURN c.id AS cid, coalesce(s.name, s.id) AS skill_name
                """,
                regex=regex,
            ).data()

            # ── Phase 2: Relation-Agnostic Indirect Multi-Hop ────
            indirect_results = session.run(
                """
                MATCH (s) WHERE s.name =~ $regex OR s.id =~ $regex
                MATCH (c_dir:Candidate)-[]->(s)
                MATCH (c_dir)-[]->(intermediate)<-[]-(c_indir:Candidate)
                WHERE NOT (c_indir)-[]->(s)
                RETURN c_indir.id AS cid, coalesce(s.name, s.id) AS skill_name, 
                       coalesce(intermediate.name, intermediate.id, 'Node') AS role_name
                LIMIT 500
                """,
                regex=regex,
            ).data()

        # Score candidates and build their traversal segments
        candidate_scores = {}
        candidate_segments = {}

        # 1. Process direct matches
        for row in direct_results:
            cid = row["cid"]
            skill = row["skill_name"]
            
            if cid not in candidate_scores:
                candidate_scores[cid] = 0.0
                candidate_segments[cid] = {}
            
            candidate_scores[cid] += 1.0  # Direct matches get full score
            candidate_segments[cid][skill] = {
                "skill": skill,
                "semantic_proxy": semantic_proxies.get(skill),
                "via_roles": [],
                "direct": True
            }

        # 2. Process indirect matches
        for row in indirect_results:
            cid = row["cid"]
            skill = row["skill_name"]
            role = row["role_name"]

            if cid not in candidate_scores:
                candidate_scores[cid] = 0.0
                candidate_segments[cid] = {}
            
            if skill not in candidate_segments[cid]:
                candidate_segments[cid][skill] = {
                    "skill": skill,
                    "semantic_proxy": semantic_proxies.get(skill),
                    "via_roles": set(),
                    "direct": False
                }
            
            # Only add via_roles if it wasn't already a direct match
            if not candidate_segments[cid][skill]["direct"]:
                candidate_segments[cid][skill]["via_roles"].add(role)

        # 3. Finalize scores and build result objects
        for cid, segs in candidate_segments.items():
            for seg in segs.values():
                if not seg["direct"]:
                    seg["via_roles"] = list(seg["via_roles"])[:2]
                    candidate_scores[cid] += 0.5  # Indirect matches get half score

        # Sort candidates by score
        sorted_cids = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)[:top_k]
        max_score = max(candidate_scores.values()) if candidate_scores else 1.0

        results = []
        for cid in sorted_cids:
            profile = self._profile_map.get(cid)
            if profile is None:
                continue

            segs = list(candidate_segments[cid].values())
            raw_score = candidate_scores[cid]
            norm_score = raw_score / max(max_score, 1.0)

            # Flat traversal list for backward compatibility
            flat_skills = [s["skill"] for s in segs]

            results.append({
                "profile": profile,
                "score": float(norm_score),
                "technique": "graph",
                "graph_traversal": flat_skills,
                "graph_traversal_detail": segs,
            })
            
        return results

    # ------------------------------------------------------------------
    # CRUD: Incremental add / remove
    # ------------------------------------------------------------------
    def add_single(self, profile: CandidateProfile):
        """Add one candidate and its relationships to the live graph."""
        self.profiles.append(profile)
        self._profile_map[profile.id] = profile

        with self.driver.session(database=self._db) as session:
            session.run(
                """
                MERGE (c:Candidate {id: $id})
                SET c.name = $name,
                    c.canonical_id = $canonical_id,
                    c.entity_type = $entity_type
                """,
                id=profile.id,
                name=profile.name or "",
                canonical_id=profile.canonical_id or "",
                entity_type=profile.entity_type or "candidate",
            )

            for attr in profile.attributes:
                if attr.value is None:
                    continue
                vals = attr.value if isinstance(attr.value, list) else [attr.value]
                label_str = attr.key.replace(" ", "").title()
                rel_type = f"HAS_{attr.key.replace(' ', '_').upper()}"
                for v in vals:
                    clean_v = str(v).strip()
                    if not clean_v:
                        continue
                    session.run(
                        f"""
                        MERGE (n:{label_str} {{name: $val}})
                        WITH n
                        MATCH (c:Candidate {{id: $id}})
                        MERGE (c)-[:{rel_type}]->(n)
                        """,
                        val=clean_v,
                        id=profile.id,
                    )
                    if clean_v not in self.node_names:
                        self.node_names.append(clean_v)

    def remove(self, profile_id: str) -> bool:
        """Remove a candidate and all its relationships from the graph."""
        if profile_id not in self._profile_map:
            return False

        with self.driver.session(database=self._db) as session:
            session.run(
                "MATCH (c:Candidate {id: $id}) DETACH DELETE c",
                id=profile_id,
            )
            # Clean orphaned nodes (nodes with no remaining connections)
            session.run(
                """
                MATCH (n)
                WHERE NOT 'Candidate' IN labels(n)
                  AND NOT (n)<-[]-()
                DELETE n
                """
            )

        del self._profile_map[profile_id]
        self.profiles = [p for p in self.profiles if p.id != profile_id]
        return True

    # ------------------------------------------------------------------
    # Graph data for visualization (unchanged)
    # ------------------------------------------------------------------
    def get_graph_data(self, max_skills: int = 60, max_candidates: int = 80) -> Dict[str, Any]:
        """Return graph data as nodes/edges dict for D3.js visualization dynamically."""
        with self.driver.session(database=self._db) as session:
            top_skills_result = session.run(
                """
                MATCH (c:Candidate)-[]->(s)
                WHERE NOT 'Candidate' IN labels(s)
                WITH s, count(c) AS degree
                ORDER BY degree DESC
                LIMIT $max_skills
                RETURN coalesce(s.name, s.id) AS name, degree
                """,
                max_skills=max_skills,
            )
            top_skills = top_skills_result.data()
            if not top_skills:
                return {"nodes": [], "edges": [], "stats": {}}

            skill_names = [r["name"] for r in top_skills if r["name"]]

            candidates_result = session.run(
                """
                MATCH (c:Candidate)-[]->(s)
                WHERE coalesce(s.name, s.id) IN $skill_names
                WITH c, count(s) AS degree
                ORDER BY degree DESC
                LIMIT $max_candidates
                RETURN c.id AS id, coalesce(c.name, c.canonical_id, c.id) AS name, degree
                """,
                skill_names=skill_names,
                max_candidates=max_candidates,
            )
            candidates = candidates_result.data()
            candidate_ids = [r["id"] for r in candidates]

            edges_result = session.run(
                """
                MATCH (c:Candidate)-[r]->(s)
                WHERE c.id IN $candidate_ids AND coalesce(s.name, s.id) IN $skill_names
                RETURN c.id AS source, coalesce(s.name, s.id) AS target, type(r) AS relation
                """,
                candidate_ids=candidate_ids,
                skill_names=skill_names,
            )
            edges = edges_result.data()

            stats_result = session.run(
                "MATCH (n) WITH coalesce(labels(n)[0], 'Entity') AS type, count(n) AS cnt RETURN type, cnt"
            )
            type_counts = {r["type"]: r["cnt"] for r in stats_result.data()}
            total_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

        nodes = []
        for s in top_skills:
            if not s["name"]: continue
            nodes.append({"id": s["name"], "label": str(s["name"]).title(), "type": "skill", "degree": s["degree"]})
        for c in candidates:
            label = (c["name"] or f"ID:{c['id']}")[:22]
            nodes.append({"id": c["id"], "label": label, "type": "candidate", "degree": c["degree"]})

        return {
            "nodes": nodes,
            "edges": [{"source": e["source"], "target": e["target"], "label": e["relation"].replace("HAS_", "").replace("_", " ").title()} for e in edges],
            "stats": {
                "total_nodes": sum(type_counts.values()),
                "total_edges": total_edges,
                "candidate_nodes": type_counts.get("Candidate", 0),
                "skill_nodes": sum(cnt for t, cnt in type_counts.items() if t != "Candidate"),
                "shown_nodes": len(nodes),
                "shown_edges": len(edges),
            },
        }
