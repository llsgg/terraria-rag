"""Split sections into chunks suitable for embedding.

Strategy:
1. Each section is a candidate chunk.
2. If a section is too long (> max chars heuristic), split it on paragraph boundaries
   with a small overlap to preserve context.
3. Each chunk carries metadata: page title, section path (for citation), pageid.

We use a *character* heuristic instead of a tokenizer here to keep the pipeline
zero-dependency at this stage; BGE-M3 truncates at `embedding_max_length` tokens
internally anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from terraria_rag.cleaning.wikitext import Section


# Roughly: BGE-M3 max_length=1024 tokens ≈ 1500-2000 Chinese chars.
# Use a conservative target.
_CHARS_PER_TOKEN_CN = 1.6


@dataclass
class Chunk:
    pageid: int
    title: str
    section_path: str   # "PageTitle > H2 > H3"
    text: str
    chunk_index: int    # within the page


def _build_section_paths(sections: list[Section]) -> list[str]:
    """For each section, build a 'a > b > c' breadcrumb based on heading levels."""
    stack: list[tuple[int, str]] = []
    paths: list[str] = []
    for s in sections:
        while stack and stack[-1][0] >= s.level:
            stack.pop()
        stack.append((s.level, s.heading))
        paths.append(" > ".join(h for _, h in stack))
    return paths


def _split_long(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split a long string on paragraph boundaries, then on sentences if still too long."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paragraphs:
        p_len = len(p)
        if cur_len + p_len + 2 <= max_chars:
            cur.append(p)
            cur_len += p_len + 2
        else:
            if cur:
                chunks.append("\n\n".join(cur))
            if p_len <= max_chars:
                cur = [p]
                cur_len = p_len
            else:
                # paragraph itself too long — hard split
                for i in range(0, p_len, max_chars - overlap_chars):
                    chunks.append(p[i : i + max_chars])
                cur, cur_len = [], 0
    if cur:
        chunks.append("\n\n".join(cur))

    # Add overlap by prepending the tail of previous chunk
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap_chars:]
            overlapped.append(tail + "\n\n" + chunks[i])
        chunks = overlapped
    return chunks


def chunk_page(
    pageid: int,
    title: str,
    sections: list[Section],
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    max_chars = int(max_tokens * _CHARS_PER_TOKEN_CN)
    overlap_chars = int(overlap_tokens * _CHARS_PER_TOKEN_CN)
    paths = _build_section_paths(sections)

    chunks: list[Chunk] = []
    idx = 0
    for sec, path in zip(sections, paths):
        body = sec.body.strip()
        if not body:
            continue
        # Prepend the section path so the model has structural context inside the chunk.
        prefix = f"# {path}\n\n"
        budget = max_chars - len(prefix)
        if budget <= 200:
            # path itself is huge, skip prefix
            prefix = ""
            budget = max_chars
        for piece in _split_long(body, budget, overlap_chars):
            chunks.append(
                Chunk(
                    pageid=pageid,
                    title=title,
                    section_path=path,
                    text=prefix + piece,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks
