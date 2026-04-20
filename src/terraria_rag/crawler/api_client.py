"""Polite MediaWiki API client with rate limiting + retries."""

from __future__ import annotations

import threading
import time
from typing import Any, Iterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from terraria_rag.config import settings


class RateLimiter:
    """Thread-safe minimum-gap limiter (good enough for polite crawling)."""

    def __init__(self, rps: float) -> None:
        self.min_gap = 1.0 / rps if rps > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_gap <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            if sleep_for > 0:
                target = self._next_allowed
            else:
                sleep_for = 0.0
                target = now
            self._next_allowed = target + self.min_gap
        if sleep_for > 0:
            time.sleep(sleep_for)


class WikiAPIClient:
    """Thin wrapper around action=query / action=parse."""

    def __init__(self) -> None:
        self.client = httpx.Client(
            base_url=settings.wiki_base_url,
            timeout=settings.crawl_timeout_sec,
            headers={
                "User-Agent": settings.wiki_user_agent,
                "Accept": "application/json",
            },
            follow_redirects=True,
        )
        self.endpoint = f"/{settings.wiki_lang}/api.php"
        self.limiter = RateLimiter(settings.crawl_rps)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "WikiAPIClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        self.limiter.wait()
        params = {**params, "format": "json", "formatversion": "2"}
        r = self.client.get(self.endpoint, params=params)
        r.raise_for_status()
        return r.json()

    # ---- enumerate pages ----

    def iter_all_pages(
        self,
        namespace: int = 0,
        apfilterredir: str = "nonredirects",
    ) -> Iterator[dict[str, Any]]:
        """Yield {pageid, ns, title} for every page in the namespace.

        ns=0 = main (articles). To grab File/Category etc., call again with other ns.
        """
        apcontinue: str | None = None
        while True:
            params: dict[str, Any] = {
                "action": "query",
                "list": "allpages",
                "apnamespace": namespace,
                "aplimit": "max",
                "apfilterredir": apfilterredir,
            }
            if apcontinue is not None:
                params["apcontinue"] = apcontinue
            data = self._get(params)
            for p in data.get("query", {}).get("allpages", []):
                yield p
            cont = data.get("continue")
            if not cont or "apcontinue" not in cont:
                return
            apcontinue = cont["apcontinue"]

    # ---- fetch wikitext ----

    @staticmethod
    def _page_to_record(page: dict[str, Any]) -> dict[str, Any] | None:
        if not page or page.get("missing"):
            return None
        revs = page.get("revisions") or []
        if not revs:
            return None
        rev = revs[0]
        wikitext = rev.get("slots", {}).get("main", {}).get("content", "")
        return {
            "pageid": page.get("pageid"),
            "title": page.get("title"),
            "revid": rev.get("revid"),
            "timestamp": rev.get("timestamp"),
            "categories": [c["title"] for c in page.get("categories", [])],
            "wikitext": wikitext,
        }

    def fetch_wikitext(self, title: str) -> dict[str, Any] | None:
        """Return {title, pageid, revid, wikitext, categories} or None if missing."""
        data = self._get(
            {
                "action": "query",
                "prop": "revisions|categories|info",
                "titles": title,
                "rvprop": "ids|content|timestamp",
                "rvslots": "main",
                "cllimit": "max",
                "redirects": 1,
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        return self._page_to_record(pages[0])

    def fetch_wikitext_batch(self, titles: list[str]) -> dict[str, dict[str, Any] | None]:
        """Fetch up to 50 titles in one request. Returns {original_title: record_or_None}.

        Resolves MediaWiki ``normalized`` and ``redirects`` rewrites so callers
        can index the result by the title they originally asked for.
        """
        if not titles:
            return {}
        if len(titles) > 50:
            raise ValueError("MediaWiki anonymous limit is 50 titles per request")

        data = self._get(
            {
                "action": "query",
                "prop": "revisions|categories|info",
                "titles": "|".join(titles),
                "rvprop": "ids|content|timestamp",
                "rvslots": "main",
                "cllimit": "max",
                "redirects": 1,
            }
        )
        query = data.get("query", {}) or {}

        # original title -> (possibly normalized) -> (possibly redirected) -> final title
        title_map: dict[str, str] = {t: t for t in titles}
        for n in query.get("normalized", []) or []:
            for orig, cur in title_map.items():
                if cur == n.get("from"):
                    title_map[orig] = n.get("to", cur)
        for r in query.get("redirects", []) or []:
            for orig, cur in title_map.items():
                if cur == r.get("from"):
                    title_map[orig] = r.get("to", cur)

        by_title: dict[str, dict[str, Any]] = {}
        for page in query.get("pages", []) or []:
            t = page.get("title")
            if t:
                by_title[t] = page

        out: dict[str, dict[str, Any] | None] = {}
        for orig in titles:
            page = by_title.get(title_map[orig])
            out[orig] = self._page_to_record(page) if page else None
        return out
