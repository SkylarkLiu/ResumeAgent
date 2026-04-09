"""
配置管理 - 基于 pydantic-settings 的环境变量加载
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，优先从 .env 读取，缺失项有默认值"""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 智谱 AI ----
    zhipuai_api_key: str = ""

    # ---- 模型 ----
    llm_model_name: str = "glm-4-flash"
    embedding_model_name: str = "embedding-3"
    vision_model_name: str = "glm-4v-flash"

    # ---- 向量检索 ----
    faiss_index_dir: str = "data/faiss_index"
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50
    kb_relevance_threshold: float = 0.35  # KB 检索最高分低于此值时降级到 web search

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    app_env: str = "development"  # development | staging | production
    debug_mode: bool = False  # 仅开发环境开启 /debug/runtime

    # ---- Checkpointer / 持久化 ----
    checkpoint_db_url: str = ""
    expert_cache_backend: str = "state_checkpointer"  # state_checkpointer | postgres
    expert_cache_db_url: str = ""
    metadata_db_url: str = ""

    # ---- 文件上传限制 ----
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"]

    # ---- Agent ----
    agent_max_history: int = 20
    agent_default_route: str = "direct"  # fallback 路由

    # ---- Web 搜索 ----
    tavily_api_key: str = ""
    web_search_max_results: int = 5
    web_search_result_max_chars: int = 500

    @property
    def faiss_index_path(self) -> Path:
        return Path(self.faiss_index_dir)

    @property
    def raw_dir(self) -> Path:
        return Path("data/raw")

    @property
    def processed_dir(self) -> Path:
        return Path("data/processed")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
