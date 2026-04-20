"""Boot the FastAPI server."""

from __future__ import annotations

import uvicorn

from terraria_rag.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "terraria_rag.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
