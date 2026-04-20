"""BGE-M3 wrapper.

BGE-M3 produces three signals per text:
- dense:    1024-dim float vector (semantic)
- sparse:   token -> weight dict (BM25-like, learned)
- colbert:  multi-vector (we don't use it here to keep storage small)

We use dense + sparse for hybrid retrieval in Qdrant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from FlagEmbedding import BGEM3FlagModel

from terraria_rag.config import settings


@dataclass
class EmbeddedChunk:
    dense: list[float]
    sparse: dict[int, float]


class BGEM3Embedder:
    DENSE_DIM = 1024

    def __init__(self) -> None:
        # use_fp16=True only on CUDA; on cpu/mps it's slower or unsupported
        use_fp16 = settings.embedding_device == "cuda"
        self.model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=use_fp16,
            devices=settings.embedding_device,
        )
        self.batch_size = settings.embedding_batch_size
        self.max_length = settings.embedding_max_length

    def encode(self, texts: Iterable[str]) -> list[EmbeddedChunk]:
        texts_list = list(texts)
        if not texts_list:
            return []
        out = self.model.encode(
            texts_list,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]                    # ndarray (N, 1024)
        lex_weights = out["lexical_weights"]         # list[dict[str, float]] — token id (str) -> weight

        results: list[EmbeddedChunk] = []
        for d, lw in zip(dense, lex_weights):
            sparse = {int(k): float(v) for k, v in lw.items() if v > 0}
            results.append(EmbeddedChunk(dense=d.tolist(), sparse=sparse))
        return results

    def encode_query(self, text: str) -> EmbeddedChunk:
        return self.encode([text])[0]
