"""Read raw pages, clean wikitext, split into chunks -> data/cleaned/chunks.jsonl"""

from __future__ import annotations

import orjson
from tqdm import tqdm

from terraria_rag.chunking.splitter import chunk_page
from terraria_rag.cleaning.wikitext import parse_to_sections
from terraria_rag.config import settings


def main() -> None:
    settings.ensure_dirs()
    pages_dir = settings.raw_dir / "pages"
    if not pages_dir.exists():
        raise SystemExit(f"missing {pages_dir} — run 02_crawl.py first")

    out_path = settings.cleaned_dir / "chunks.jsonl"
    files = sorted(pages_dir.glob("*.json"))
    if not files:
        raise SystemExit("no cached pages found")

    n_chunks = 0
    n_pages = 0
    with open(out_path, "wb") as out:
        for fp in tqdm(files, desc="cleaning", unit="pg"):
            with open(fp, "rb") as f:
                page = orjson.loads(f.read())
            try:
                sections = parse_to_sections(page["title"], page.get("wikitext") or "")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] parse failed for {page['title']}: {e}")
                continue
            chunks = chunk_page(
                pageid=int(page["pageid"]),
                title=str(page["title"]),
                sections=sections,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            for ch in chunks:
                out.write(
                    orjson.dumps(
                        {
                            "pageid": ch.pageid,
                            "title": ch.title,
                            "section_path": ch.section_path,
                            "chunk_index": ch.chunk_index,
                            "text": ch.text,
                        }
                    )
                )
                out.write(b"\n")
            n_chunks += len(chunks)
            n_pages += 1
    print(f"[done] {n_pages} pages -> {n_chunks} chunks -> {out_path}")


if __name__ == "__main__":
    main()
