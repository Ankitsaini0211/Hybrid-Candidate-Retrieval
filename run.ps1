Write-Host "============================================"
Write-Host "  Dynamic Semantic Retrieval System"
Write-Host "============================================"
Write-Host ""

Write-Host "Step 1: Installing Python dependencies..."
pip install fastapi "uvicorn[standard]" pydantic torch sentence-transformers faiss-cpu networkx pandas rank_bm25 nltk python-multipart neo4j
Write-Host ""

Write-Host "Step 2: Checking Neo4j connection..."
python -c "from neo4j import GraphDatabase; d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','Akshat007')); d.verify_connectivity(); print('Neo4j connected!'); d.close()"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[WARNING] Neo4j not reachable at bolt://localhost:7687"
    Write-Host "  The system will start WITHOUT graph features."
    Write-Host "  To enable: start Neo4j Desktop or run:"
    Write-Host "    docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/Akshat007 neo4j:latest"
    Write-Host ""
}

Write-Host "Step 3: Starting application..."
Write-Host "  Server will be at: http://localhost:8000"
Write-Host ""
python main.py
