"""Fetch wikitext for every page in page_index.jsonl -> data/raw/pages/{pageid}.json

Idempotent: skips pageids already on disk. Safe to interrupt and resume.

Speedups vs the naive 1-title-per-request loop:
  * batch up to 50 titles per MediaWiki call
  * run several batches concurrently via a thread pool (RateLimiter is shared)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
from tqdm import tqdm

from terraria_rag.config import settings
from terraria_rag.crawler.api_client import WikiAPIClient

MEDIAWIKI_TITLES_HARD_LIMIT = 50  # anonymous limit per request


def _load_index(index_path: Path) -> list[tuple[int, str]]:
    titles: list[tuple[int, str]] = []
    with open(index_path, "rb") as f:
        for line in f:
            entry = orjson.loads(line)
            titles.append((int(entry["pageid"]), str(entry["title"])))
    return titles


def _chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=settings.crawl_batch_size,
        help=f"titles per request (max {MEDIAWIKI_TITLES_HARD_LIMIT})",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=settings.crawl_concurrency,
        help="parallel in-flight batches (RateLimiter still applies globally)",
    )
    args = ap.parse_args()

    batch_size = max(1, min(args.batch_size, MEDIAWIKI_TITLES_HARD_LIMIT))
    concurrency = max(1, args.concurrency)

    settings.ensure_dirs()
    pages_dir = settings.raw_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    index_path = settings.raw_dir / "page_index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path} — run 01_enumerate.py first")

    titles = _load_index(index_path)

    todo: list[tuple[int, str, Path]] = []
    for pid, title in titles:
        out = pages_dir / f"{pid}.json"
        if out.exists() and not args.force:
            continue
        todo.append((pid, title, out))

    print(
        f"[plan] {len(titles)} total | {len(titles) - len(todo)} cached | "
        f"{len(todo)} to fetch | batch={batch_size} concurrency={concurrency} "
        f"rps={settings.crawl_rps}"
    )
    if not todo:
        print("[done] nothing to do")
        return

    title_to_target: dict[str, tuple[int, Path]] = {t: (pid, out) for pid, t, out in todo}
    batches: list[list[str]] = list(_chunked([t for _, t, _ in todo], batch_size))

    failed: list[tuple[int, str, str]] = []
    written = 0

    def _process_batch(client: WikiAPIClient, batch: list[str]) -> tuple[int, list[tuple[int, str, str]]]:
        local_failed: list[tuple[int, str, str]] = []
        try:
            results = client.fetch_wikitext_batch(batch)
        except Exception as e:  # noqa: BLE001
            for t in batch:
                pid, _ = title_to_target[t]
                local_failed.append((pid, t, repr(e)))
            return 0, local_failed

        local_written = 0
        for t in batch:
            pid, out = title_to_target[t]
            data = results.get(t)
            if data is None:
                local_failed.append((pid, t, "missing or no revisions"))
                continue
            try:
                tmp = out.with_suffix(out.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    f.write(orjson.dumps(data))
                tmp.replace(out)
                local_written += 1
            except OSError as e:
                local_failed.append((pid, t, f"write error: {e!r}"))
        return local_written, local_failed

    with WikiAPIClient() as client, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_process_batch, client, b) for b in batches]
        with tqdm(total=len(todo), desc="fetching", unit="pg") as pbar:
            for fut in as_completed(futures):
                w, fails = fut.result()
                written += w
                failed.extend(fails)
                pbar.update(w + len(fails))

    if failed:
        fail_path = settings.raw_dir / "failures.jsonl"
        with open(fail_path, "wb") as f:
            for pid, title, err in failed:
                f.write(orjson.dumps({"pageid": pid, "title": title, "error": err}))
                f.write(b"\n")
        print(f"\n[warn] {len(failed)} failures logged -> {fail_path}")
    print(f"[done] wrote {written} pages -> {pages_dir}")


if __name__ == "__main__":
    main()
