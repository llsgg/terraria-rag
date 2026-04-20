"""Polite MediaWiki API client with rate limiting + retries."""

from __future__ import annotations

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
    """Token-bucket-ish: enforce minimum gap between requests."""

    def __init__(self, rps: float) -> None:
        self.min_gap = 1.0 / rps if rps > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_gap <= 0:
            return
        gap = time.monotonic() - self._last
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self._last = time.monotonic()


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
        page = pages[0]
        if page.get("missing"):
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
