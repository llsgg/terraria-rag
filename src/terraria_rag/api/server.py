"""FastAPI server exposing /query for retrieval.

This is a *retrieval-only* endpoint by default — it returns the top chunks with
citations so any frontend / LLM can take it from there. We deliberately don't
hard-wire a specific LLM call so users can plug in OpenAI / DeepSeek / local
Ollama / 通义 / 智谱 etc. on top.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from terraria_rag.config import settings
from terraria_rag.embedding.bge import BGEM3Embedder
from terraria_rag.store.qdrant_store import QdrantStore, RetrievedChunk


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class QueryHit(BaseModel):
    score: float
    title: str
    section_path: str
    url: str
    text: str


class QueryResponse(BaseModel):
    query: str
    hits: list[QueryHit]
    total_chunks: int


_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    print(f"[boot] Loading BGE-M3 from {settings.embedding_model} on {settings.embedding_device}...")
    _state["embedder"] = BGEM3Embedder()
    print("[boot] Connecting Qdrant at", settings.qdrant_path)
    _state["store"] = QdrantStore()
    print("[boot] Ready. Indexed chunks:", _state["store"].count())
    yield
    _state["store"].close()


app = FastAPI(title="Terraria Wiki RAG", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    store: QdrantStore = _state["store"]  # type: ignore[assignment]
    return {"status": "ok", "indexed_chunks": store.count()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    embedder: BGEM3Embedder = _state["embedder"]  # type: ignore[assignment]
    store: QdrantStore = _state["store"]          # type: ignore[assignment]
    top_k = req.top_k or settings.retrieval_top_k

    q_emb = embedder.encode_query(req.query)
    try:
        hits: list[RetrievedChunk] = store.hybrid_search(q_emb.dense, q_emb.sparse, top_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"search failed: {e}") from e

    return QueryResponse(
        query=req.query,
        total_chunks=store.count(),
        hits=[
            QueryHit(
                score=h.score,
                title=h.title,
                section_path=h.section_path,
                url=h.url,
                text=h.text,
            )
            for h in hits
        ],
    )
