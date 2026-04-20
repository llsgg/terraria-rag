"""List all wiki page titles -> data/raw/page_index.jsonl

Usage:
    uv run python scripts/01_enumerate.py            # all main-namespace pages
    uv run python scripts/01_enumerate.py --limit 50 # smoke test
"""

from __future__ import annotations

import argparse

import orjson
from tqdm import tqdm

from terraria_rag.config import settings
from terraria_rag.crawler.api_client import WikiAPIClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N pages")
    ap.add_argument("--namespace", type=int, default=0, help="MediaWiki namespace id")
    args = ap.parse_args()

    settings.ensure_dirs()
    out_path = settings.raw_dir / "page_index.jsonl"

    n = 0
    with WikiAPIClient() as client, open(out_path, "wb") as f:
        bar = tqdm(desc="enumerating pages", unit="pg")
        for page in client.iter_all_pages(namespace=args.namespace):
            f.write(orjson.dumps(page))
            f.write(b"\n")
            n += 1
            bar.update(1)
            if args.limit and n >= args.limit:
                break
        bar.close()
    print(f"\n[done] wrote {n} page entries -> {out_path}")


if __name__ == "__main__":
    main()
