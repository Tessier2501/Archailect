"""集中管理环境变量, 并提供 LLM / Embedding 工厂函数。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

load_dotenv()

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent          # Archailect/
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}, 请检查 .env")
    return value


# ---- LLM (DeepSeek) ----
DEEPSEEK_API_KEY = _require_env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---- Embedding (第三方 OpenAI 兼容 API; 框架阶段为占位值) ----
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192"))


def build_embedding_func() -> EmbeddingFunc:
    """构造 OpenAI 兼容的 embedding 函数。

    设计决策 (用户确认): 1.5.6 在实例化即强制校验 embedding_func (不允许 None),
    因此本函数始终返回 EmbeddingFunc。框架阶段 .env 的 EMBEDDING_* 为占位值,
    使 LightRAG 可正常实例化、服务可启动; 占位配置下误触发建图会得到 OpenAI API
    的明确报错 (Fail-Fast), 不会静默产生假数据。真实建图前必须将 EMBEDDING_*
    替换为有效配置; 更换 embedding 模型后须重建全部索引 (见 plan.md §8 风险表)。
    """
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        )(openai_embed),
    )