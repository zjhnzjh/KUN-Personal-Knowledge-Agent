from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .credentials import get_secret


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    dashscope_api_key: str
    dashscope_base_url: str
    embedding_model: str
    embedding_dimension: int
    dashscope_rerank_base_url: str
    rerank_model: str
    vision_model: str


def get_settings() -> Settings:
    configured = os.getenv("KUN_DATA_DIR", "").strip()
    local = os.getenv("LOCALAPPDATA", str(Path.home()))
    data_dir = Path(configured) if configured else Path(local) / "KUN"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=data_dir,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip() or get_secret("deepseek"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", "").strip() or get_secret("dashscope"),
        dashscope_base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/"),
        embedding_model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "qwen3.7-text-embedding"),
        embedding_dimension=int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSION", "1024")),
        dashscope_rerank_base_url=os.getenv("DASHSCOPE_RERANK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-api/v1").rstrip("/"),
        rerank_model=os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank"),
        vision_model=os.getenv("DASHSCOPE_VISION_MODEL", "qwen3-vl-flash"),
    )
