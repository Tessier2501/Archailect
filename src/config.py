"""集中管理环境变量, 并提供 LLM / Embedding 工厂函数."""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

# override=True: .env 是权威配置源. 默认不覆盖会导致 shell 残留同名
# 环境变量 (如 source .env 后的旧值) 遮蔽 .env 的新配置, 已实测踩坑.
load_dotenv(override=True)

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent          # Archailect/
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

# ---- 缓存 ----
RAG_CACHE_MAX = int(os.getenv("RAG_CACHE_MAX", "8"))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}, 请检查 .env")
    return value


# ---- LLM (ds v4 flash, cherryin 网关) ----
DEEPSEEK_API_KEY = _require_env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://open.cherryin.ai/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-v4-flash")


async def _llm_wrapper(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """LightRAG 可用的 LLM 包装函数 (显式参数对齐).

    核心背景: LightRAG 以 llm_model_func(prompt, system_prompt=..., **kwargs)
    位置调用, 而 openai_complete_if_cache 的第一位置参数是 model.
    裸传/llm_model_kwargs/partial 绑定 model 都会导致参数错位.
    本 wrapper 显式接收 LightRAG 的位置参数 (prompt), 再以关键字转调
    openai_complete_if_cache, 使 provider 配置与传入参数全部正确对齐.
    """
    return await openai_complete_if_cache(
        prompt=prompt,
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


def build_llm_func():
    """返回 LightRAG 可直接用作 llm_model_func 的绑定函数."""
    return partial(_llm_wrapper)

# ---- Embedding (Qwen3-Embedding-8B, cherryin 网关) ----
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://open.cherryin.ai/v1")
# LLM 与 Embedding 共用同一 cherryin key; EMBEDDING_API_KEY 缺省时回退 DEEPSEEK_API_KEY.
# 独立设置 EMBEDDING_API_KEY 仅用于未来 embedding 与 LLM 走不同 provider.
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", DEEPSEEK_API_KEY)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "32768"))


def build_embedding_func() -> EmbeddingFunc:
    """构造 OpenAI 兼容的 embedding 函数.

    lightrag-hku 1.5.6 在实例化即强制校验 embedding_func (不允许 None).
    注意:
    1. EmbeddingFunc 只按 func(texts) 调用, 无 kwargs 传递机制, 必须用
       functools.partial 预绑定 model/base_url/api_key, 否则 openai_embed
       回退读环境变量 OPENAI_API_KEY 而 KeyError.
    2. openai_embed 本身已是装饰后的 EmbeddingFunc 实例 (默认 dim=1536).
       直接包装会被 EmbeddingFunc.__post_init__ unwrap 成内层 1536.
       必须用 openai_embed.func 取其原始未装饰函数再包装.
    """
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        )(
            partial(
                openai_embed.func,  # 原始函数, 避免嵌套 EmbeddingFunc 被 unwrap
                model=EMBEDDING_MODEL,
                base_url=EMBEDDING_BASE_URL,
                api_key=EMBEDDING_API_KEY,
            )
        ),
    )