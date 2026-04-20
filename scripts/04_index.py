"""Embed chunks with BGE-M3 and upsert into Qdrant.

Usage:
    uv run python scripts/04_index.py             # incremental (skips existing collection)
    uv run python scripts/04_index.py --rebuild   # drop & recreate collection
"""

from __future__ import annotations

# IMPORTANT: load .env *before* any huggingface / transformers / torch import
# so HF_ENDPOINT (HF mirror) is honored when files are first resolved.
import os
from dotenv import load_dotenv

load_dotenv(override=False)
# Belt-and-suspenders: hf_hub also looks at HF_HUB_ENDPOINT (newer envs)
if os.getenv("HF_ENDPOINT") and not os.getenv("HF_HUB_ENDPOINT"):
    os.environ["HF_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]

import argparse
from itertools import islice
from typing import Iterator

import orjson
from tqdm import tqdm

from terraria_rag.chunking.splitter import Chunk
from terraria_rag.config import settings
from terraria_rag.embedding.bge import BGEM3Embedder
from terraria_rag.store.qdrant_store import QdrantStore


def _iter_chunks(path) -> Iterator[Chunk]:
    with open(path, "rb") as f:
        for line in f:
            d = orjson.loads(line)
            yield Chunk(
                pageid=int(d["pageid"]),
                title=str(d["title"]),
                section_path=str(d["section_path"]),
                text=str(d["text"]),
                chunk_index=int(d["chunk_index"]),
            )


def _batched(it, n):
    it = iter(it)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="drop & recreate the collection")
    ap.add_argument("--batch-size", type=int, default=32, help="chunks per upsert batch")
    args = ap.parse_args()

    settings.ensure_dirs()
    chunks_path = settings.cleaned_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"missing {chunks_path} — run 03_clean_chunk.py first")

    total = sum(1 for _ in open(chunks_path, "rb"))
    print(f"[plan] {total} chunks to index")

    print(f"[boot] loading BGE-M3 ({settings.embedding_model}, device={settings.embedding_device})")
    embedder = BGEM3Embedder()

    print(f"[boot] opening qdrant at {settings.qdrant_path}")
    store = QdrantStore()
    store.ensure_collection(dense_dim=BGEM3Embedder.DENSE_DIM, recreate=args.rebuild)

    bar = tqdm(total=total, desc="indexing", unit="chk")
    try:
        for batch in _batched(_iter_chunks(chunks_path), args.batch_size):
            embs = embedder.encode([c.text for c in batch])
            store.upsert(batch, embs)
            bar.update(len(batch))
    finally:
        bar.close()
        store.close()
    print(f"[done] indexed -> collection '{settings.qdrant_collection}'")


if __name__ == "__main__":
    main()
