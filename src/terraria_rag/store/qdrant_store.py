"""Qdrant store: hybrid (dense + sparse) collection backed by a local file.

We use the embedded mode (`path=...`) so no docker / server is required.
The same code works against a remote Qdrant by swapping the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from terraria_rag.chunking.splitter import Chunk
from terraria_rag.config import settings
from terraria_rag.embedding.bge import EmbeddedChunk

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"


@dataclass
class RetrievedChunk:
    score: float
    pageid: int
    title: str
    section_path: str
    text: str
    url: str


def _page_url(title: str) -> str:
    safe = title.replace(" ", "_")
    return f"{settings.page_url_prefix}/{safe}"


class QdrantStore:
    def __init__(self) -> None:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(settings.qdrant_path))
        self.collection = settings.qdrant_collection

    # ---- schema ----

    def ensure_collection(self, dense_dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE_NAME: qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE),
                },
                sparse_vectors_config={
                    SPARSE_NAME: qm.SparseVectorParams(),
                },
            )

    # ---- write ----

    def upsert(self, chunks: list[Chunk], embeddings: list[EmbeddedChunk]) -> None:
        assert len(chunks) == len(embeddings)
        points: list[qm.PointStruct] = []
        for ch, emb in zip(chunks, embeddings):
            pid = self._point_id(ch.pageid, ch.chunk_index)
            payload: dict[str, Any] = {
                "pageid": ch.pageid,
                "title": ch.title,
                "section_path": ch.section_path,
                "chunk_index": ch.chunk_index,
                "text": ch.text,
                "url": _page_url(ch.title),
            }
            sparse_indices = list(emb.sparse.keys())
            sparse_values = list(emb.sparse.values())
            points.append(
                qm.PointStruct(
                    id=pid,
                    vector={
                        DENSE_NAME: emb.dense,
                        SPARSE_NAME: qm.SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                    },
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    @staticmethod
    def _point_id(pageid: int, chunk_index: int) -> int:
        # Pack (pageid, chunk_index) into one stable 64-bit id.
        # Assumes chunk_index < 4096 per page (>>1 enough for our pages).
        return (int(pageid) << 12) | (int(chunk_index) & 0xFFF)

    # ---- read ----

    def hybrid_search(
        self,
        dense: list[float],
        sparse: dict[int, float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Use Qdrant's Query API to do RRF fusion of dense + sparse."""
        prefetch_k = max(top_k * 4, 32)
        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(
                    query=dense,
                    using=DENSE_NAME,
                    limit=prefetch_k,
                ),
                qm.Prefetch(
                    query=qm.SparseVector(
                        indices=list(sparse.keys()),
                        values=list(sparse.values()),
                    ),
                    using=SPARSE_NAME,
                    limit=prefetch_k,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        out: list[RetrievedChunk] = []
        for p in result.points:
            payload = p.payload or {}
            out.append(
                RetrievedChunk(
                    score=float(p.score),
                    pageid=int(payload.get("pageid", 0)),
                    title=str(payload.get("title", "")),
                    section_path=str(payload.get("section_path", "")),
                    text=str(payload.get("text", "")),
                    url=str(payload.get("url", "")),
                )
            )
        return out

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def close(self) -> None:
        self.client.close()
