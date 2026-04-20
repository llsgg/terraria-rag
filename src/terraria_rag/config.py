"""Global configuration loaded from env / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ as early as possible so third-party libs
# (huggingface_hub, torch, etc.) that read env at import time also see it.
load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    wiki_base_url: str = "https://terraria.wiki.gg"
    wiki_lang: str = "zh"
    wiki_user_agent: str = "terraria-rag-bot/0.1 (personal study)"

    crawl_rps: float = 1.0
    crawl_timeout_sec: int = 30

    data_dir: Path = Path("./data")
    qdrant_path: Path = Path("./data/qdrant")
    qdrant_collection: str = "terraria_zh"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    embedding_max_length: int = 1024

    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    retrieval_top_k: int = 8

    @property
    def api_endpoint(self) -> str:
        return f"{self.wiki_base_url}/{self.wiki_lang}/api.php"

    @property
    def page_url_prefix(self) -> str:
        return f"{self.wiki_base_url}/{self.wiki_lang}/wiki"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cleaned_dir(self) -> Path:
        return self.data_dir / "cleaned"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.raw_dir, self.cleaned_dir, self.qdrant_path):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
