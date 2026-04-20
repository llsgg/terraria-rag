"""Embed chunks with BGE-M3 and upsert into Qdrant.

Resumable: on restart, the script scrolls existing point ids from the
collection and skips chunks that are already indexed, so the costly BGE-M3
embedding step is only run for new chunks.

Usage:
    uv run python scripts/04_index.py                  # incremental, resume-aware
    uv run python scripts/04_index.py --rebuild        # drop & recreate, then index all
    uv run python scripts/04_index.py --force-reindex  # re-embed even existing ids
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
    ap.add_argument(
        "--force-reindex",
        action="store_true",
        help="re-embed and upsert every chunk even if its id already exists",
    )
    ap.add_argument("--batch-size", type=int, default=32, help="chunks per upsert batch")
    args = ap.parse_args()

    settings.ensure_dirs()
    chunks_path = settings.cleaned_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(f"missing {chunks_path} — run 03_clean_chunk.py first")

    total = sum(1 for _ in open(chunks_path, "rb"))
    print(f"[plan] {total} chunks in {chunks_path}")

    print(f"[boot] opening qdrant at {settings.qdrant_path}")
    store = QdrantStore()
    store.ensure_collection(dense_dim=BGEM3Embedder.DENSE_DIM, recreate=args.rebuild)

    # Resume support: figure out which point ids are already indexed.
    # `--rebuild` just dropped the collection, so the set is trivially empty.
    # `--force-reindex` ignores the set and re-embeds everything.
    if args.rebuild or args.force_reindex:
        existing: set[int] = set()
        if args.force_reindex and not args.rebuild:
            print("[resume] --force-reindex: re-embedding every chunk")
    else:
        print("[resume] scanning existing point ids in collection ...")
        existing = store.existing_ids()
        print(f"[resume] {len(existing)} chunks already indexed, will be skipped")

    # Lazy-load the embedder: if there's nothing left to do we can avoid the
    # multi-second BGE-M3 startup entirely.
    embedder: BGEM3Embedder | None = None

    def _get_embedder() -> BGEM3Embedder:
        nonlocal embedder
        if embedder is None:
            print(
                f"[boot] loading BGE-M3 ({settings.embedding_model}, "
                f"device={settings.embedding_device})"
            )
            embedder = BGEM3Embedder()
        return embedder

    skipped = 0
    indexed = 0
    bar = tqdm(total=total, desc="indexing", unit="chk")
    try:
        for raw_batch in _batched(_iter_chunks(chunks_path), args.batch_size):
            # Filter out chunks that are already in the collection.
            todo: list[Chunk] = []
            for ch in raw_batch:
                pid = QdrantStore._point_id(ch.pageid, ch.chunk_index)
                if pid in existing:
                    skipped += 1
                else:
                    todo.append(ch)
            if todo:
                embs = _get_embedder().encode([c.text for c in todo])
                store.upsert(todo, embs)
                indexed += len(todo)
            bar.update(len(raw_batch))
            bar.set_postfix(indexed=indexed, skipped=skipped, refresh=False)
    finally:
        bar.close()
        store.close()
    print(
        f"[done] collection '{settings.qdrant_collection}': "
        f"indexed={indexed} skipped={skipped} total={total}"
    )


if __name__ == "__main__":
    main()
