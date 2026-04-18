"""
FastAPI application — rewired for the Dynamic Semantic Retrieval pipeline.

Pipeline: Query → QueryAnalyzer → [BM25 + HNSW] → Weighted RRF → Top-K
          → Neo4j Graph Features → ColBERT Re-rank → Final Score → Answer
"""
# Load .env FIRST so all os.getenv() calls pick up Aura credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import time

from src.core.config import SystemConfig
from src.core.data_loader import load_and_clean_data
from src.core.embedding_service import EmbeddingService
from src.core.feature_pipeline import FeaturePipeline
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.vector_indexer import VectorIndexer
from src.indexing.graph_indexer import GraphIndexer
from src.core.graph_features import GraphFeatureExtractor
from src.core.reranker import ColBERTReranker, NoopReranker
from src.core.retriever import HybridRetriever

app = FastAPI(title="Dynamic Semantic Retrieval System")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Global state ───────────────────────────────────────────────────────
config = SystemConfig()
retriever: HybridRetriever | None = None
graph_indexer: GraphIndexer | None = None
embedding_service: EmbeddingService | None = None
feature_pipeline: FeaturePipeline | None = None


@app.on_event("startup")
async def startup_event():
    global retriever, graph_indexer, embedding_service, feature_pipeline

    t0 = time.time()

    # ── Resolve data path ──────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    candidates = [
        os.path.join(base_dir, "data", "profiles.csv"),
        os.path.join(base_dir, "..", "profiles.csv"),
        os.path.join(base_dir, "profiles.csv"),
    ]
    data_path = next((c for c in candidates if os.path.exists(c)), None)
    if data_path is None:
        raise FileNotFoundError("profiles.csv not found. Searched:\n" + "\n".join(candidates))

    print(f"Loading profiles from: {data_path}")
    profiles = load_and_clean_data(data_path)

    # ── Shared services ────────────────────────────────────────────────
    print("Initializing shared embedding service...")
    embedding_service = EmbeddingService(model_name=config.embedding_model)
    feature_pipeline = FeaturePipeline()

    # ── Stage 1 indexers ───────────────────────────────────────────────
    print("Initializing BM25 indexer...")
    bm25 = BM25Indexer()
    bm25.set_pipeline(feature_pipeline)
    bm25.index(profiles)

    print(f"Initializing HNSW vector indexer ({config.index_type})...")
    vector = VectorIndexer(embedding_service, feature_pipeline, config)
    vector.index_all(profiles)

    # ── Neo4j (feature layer only) ─────────────────────────────────────
    graph_features = None
    try:
        print("Initializing Neo4j graph indexer (feature layer)...")
        graph_indexer = GraphIndexer()
        graph_indexer.index(profiles, embedding_service=embedding_service)
        graph_features = GraphFeatureExtractor(graph_indexer)
        print("Neo4j graph features ready.")
    except Exception as e:
        print(f"[WARNING] Neo4j unavailable ({e}). Graph features disabled.")
        graph_features = None

    # ── Stage 2 reranker ───────────────────────────────────────────────
    if config.enable_colbert_rerank:
        print("Initializing ColBERT re-ranker...")
        reranker = ColBERTReranker(embedding_service, feature_pipeline)
    else:
        reranker = NoopReranker()

    # ── Assemble pipeline ──────────────────────────────────────────────
    retriever = HybridRetriever(
        bm25=bm25,
        vector=vector,
        graph_features=graph_features,
        reranker=reranker,
        profiles=profiles,
        config=config,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Dynamic Semantic Retrieval System — READY")
    print(f"  Profiles:    {len(profiles)}")
    print(f"  Index:       {config.index_type.upper()} (M={config.hnsw_m}, efSearch={config.hnsw_ef_search})")
    print(f"  Reranker:    {'ColBERT (MaxSim)' if config.enable_colbert_rerank else 'Disabled'}")
    print(f"  Graph:       {'Neo4j features' if graph_features else 'Disabled'}")
    print(f"  Weights:     {'Dynamic (intent-adaptive)' if config.dynamic_weighting else 'Static'}")
    print(f"  Startup:     {elapsed:.1f}s")
    print(f"{'='*60}\n")


@app.on_event("shutdown")
async def shutdown_event():
    if graph_indexer:
        graph_indexer.close()


# ── Search ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None, "query": ""})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), limit: str = Form("10")):
    top_k = 99999 if limit == "max" else int(limit)
    t0 = time.time()
    results, intent_info = retriever.search(query, top_k=top_k)
    latency = (time.time() - t0) * 1000  # ms
    intent_info["latency_ms"] = f"{latency:.0f}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "query": query,
            "intent": intent_info,
            "limit": limit,
        },
    )


# ── Graph visualization ───────────────────────────────────────────────
@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request):
    return templates.TemplateResponse("graph.html", {"request": request})


@app.get("/graph-data")
async def graph_data(max_skills: int = 50, max_candidates: int = 60):
    """Return graph nodes/edges from Neo4j for D3.js visualization."""
    if graph_indexer is None:
        return JSONResponse({"nodes": [], "edges": []})
    data = graph_indexer.get_graph_data(max_skills=max_skills, max_candidates=max_candidates)
    return JSONResponse(data)


# ── API: Stats ─────────────────────────────────────────────────────────
@app.get("/api/stats")
async def stats():
    return JSONResponse({
        "config": {
            "index_type": config.index_type,
            "hnsw_m": config.hnsw_m,
            "hnsw_ef_search": config.hnsw_ef_search,
            "colbert_enabled": config.enable_colbert_rerank,
            "dynamic_weights": config.dynamic_weighting,
            "graph_features": graph_indexer is not None,
        },
        "vector_index": retriever.vector.stats if retriever else {},
    })


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
