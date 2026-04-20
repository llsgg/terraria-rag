"""Fetch wikitext for every page in page_index.jsonl -> data/raw/pages/{pageid}.json

Idempotent: skips pageids already on disk. Safe to interrupt and resume.
"""

from __future__ import annotations

import argparse

import orjson
from tqdm import tqdm

from terraria_rag.config import settings
from terraria_rag.crawler.api_client import WikiAPIClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    settings.ensure_dirs()
    pages_dir = settings.raw_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    index_path = settings.raw_dir / "page_index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path} — run 01_enumerate.py first")

    titles: list[tuple[int, str]] = []
    with open(index_path, "rb") as f:
        for line in f:
            entry = orjson.loads(line)
            titles.append((int(entry["pageid"]), str(entry["title"])))

    todo = []
    for pid, title in titles:
        out = pages_dir / f"{pid}.json"
        if out.exists() and not args.force:
            continue
        todo.append((pid, title, out))

    print(f"[plan] {len(titles)} total | {len(titles) - len(todo)} cached | {len(todo)} to fetch")

    if not todo:
        print("[done] nothing to do")
        return

    failed: list[tuple[int, str, str]] = []
    with WikiAPIClient() as client:
        for pid, title, out in tqdm(todo, desc="fetching", unit="pg"):
            try:
                data = client.fetch_wikitext(title)
            except Exception as e:  # noqa: BLE001
                failed.append((pid, title, repr(e)))
                continue
            if data is None:
                failed.append((pid, title, "missing or no revisions"))
                continue
            with open(out, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    if failed:
        fail_path = settings.raw_dir / "failures.jsonl"
        with open(fail_path, "wb") as f:
            for pid, title, err in failed:
                f.write(orjson.dumps({"pageid": pid, "title": title, "error": err}))
                f.write(b"\n")
        print(f"\n[warn] {len(failed)} failures logged -> {fail_path}")
    print(f"[done] cached pages -> {pages_dir}")


if __name__ == "__main__":
    main()
