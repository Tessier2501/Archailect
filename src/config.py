"""集中管理环境变量, 并提供 LLM / Embedding 工厂函数.

架构 (2026-08-11 调整): LLM 与 Embedding 均统一走 Cherry 网关.
免费优先 (LLM_FREE_MODEL / EMBEDDING_MODEL_FREE), 被 429/限流后
触发付费兜底 (LLM_PAID_MODEL / EMBEDDING_MODEL_PAID), 冷却
FALLBACK_COOLDOWN 秒内直接走付费, 冷却过后自动回免费.
"""
from __future__ import annotations

import os
import threading
import time
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


def _resolve_env(name: str) -> str:
    """读取环境变量并拒绝未填写的占位符 (<...>)."""
    value = _require_env(name)
    if "<" in value or ">" in value or value.startswith("$"):
        raise RuntimeError(
            f"环境变量 {name} 仍是占位符, 请先在 .env 填入真实值 "
            f"(当前值: {value!r})"
        )
    return value


# ============================================================
# Cherry 统一网关配置 (LLM 与 Embedding 共用 base_url/api_key)
# ============================================================
CHERRY_BASE_URL = _resolve_env("CHERRY_BASE_URL")
CHERRY_API_KEY = _resolve_env("CHERRY_API_KEY")

LLM_FREE_MODEL = _resolve_env("LLM_FREE_MODEL")
LLM_PAID_MODEL = _resolve_env("LLM_PAID_MODEL")
EMBEDDING_MODEL_FREE = _resolve_env("EMBEDDING_MODEL_FREE")
EMBEDDING_MODEL_PAID = _resolve_env("EMBEDDING_MODEL_PAID")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "32768"))
FALLBACK_COOLDOWN = float(os.getenv("FALLBACK_COOLDOWN", "60"))


# ============================================================
# 免费/付费双档状态机 (线程安全)
# ============================================================
class _FallbackCircuit:
    """免费优先; 429/限流触发切付费并进入冷却; 冷却后自动回免费.

    - should_use_paid(): 当前是否应直接走付费 (冷却期内)
    - trip(): 标记免费被限流, 进入 FALLBACK_COOLDOWN 秒付费冷却
    - 用 threading.Lock 保证 Embedding worker 多线程调用下安全
    """

    def __init__(self, cooldown: float) -> None:
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._paid_until: float = 0.0

    def should_use_paid(self) -> bool:
        with self._lock:
            return time.monotonic() < self._paid_until

    def trip(self) -> None:
        with self._lock:
            self._paid_until = time.monotonic() + self._cooldown


_LLM_FALLBACK = _FallbackCircuit(FALLBACK_COOLDOWN)
_EMBED_FALLBACK = _FallbackCircuit(FALLBACK_COOLDOWN)


# ============================================================
# LLM (统一 cherry, 免费优先 + 付费兜底)
# ============================================================
async def _llm_wrapper(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """LightRAG 可用的 LLM 包装函数.

    核心背景: LightRAG 以 llm_model_func(prompt, system_prompt=..., **kwargs)
    位置调用, 而 openai_complete_if_cache 的第一位置参数是 model.
    本 wrapper 显式接收位置参数, 再以关键字转调, 避免参数错位.
    free/paid 切换: 免费优先; 免费重试耗尽仍 429/限流则触发付费冷却并兜底.
    """
    use_paid = _LLM_FALLBACK.should_use_paid()
    model = LLM_PAID_MODEL if use_paid else LLM_FREE_MODEL
    kwargs_for_call = {
        "prompt": prompt,
        "model": model,
        "base_url": CHERRY_BASE_URL,
        "api_key": CHERRY_API_KEY,
        "system_prompt": system_prompt,
        "history_messages": history_messages,
        **kwargs,
    }
    try:
        return await openai_complete_if_cache(**kwargs_for_call)
    except Exception:
        # openai SDK 已对 429 内建重试; 这里捕获的是重试耗尽后的限流类异常.
        # 触发付费冷却, 并用付费模型重试一次兜底.
        _LLM_FALLBACK.trip()
        kwargs_for_call["model"] = LLM_PAID_MODEL
        return await openai_complete_if_cache(**kwargs_for_call)


def build_llm_func():
    """返回 LightRAG 可直接用作 llm_model_func 的绑定函数."""
    return partial(_llm_wrapper)


# ============================================================
# Embedding (统一 cherry, 免费优先 + 付费兜底)
# ============================================================
async def _embed_with_fallback(texts: list[str]) -> list[list[float]]:
    """Embedding 免费优先, 429/限流触发付费冷却并兜底, 维度恒定 1024."""
    use_paid = _EMBED_FALLBACK.should_use_paid()
    model = EMBEDDING_MODEL_PAID if use_paid else EMBEDDING_MODEL_FREE
    try:
        return await openai_embed.func(
            model=model,
            base_url=CHERRY_BASE_URL,
            api_key=CHERRY_API_KEY,
            texts=texts,
        )
    except Exception:
        # openai SDK 已对 429 内建重试; 此处为重试耗尽后的限流兜底.
        _EMBED_FALLBACK.trip()
        return await openai_embed.func(
            model=EMBEDDING_MODEL_PAID,
            base_url=CHERRY_BASE_URL,
            api_key=CHERRY_API_KEY,
            texts=texts,
        )


def build_embedding_func() -> EmbeddingFunc:
    """构造带免费/付费兜底的 embedding 函数.

    lightrag-hku 1.5.6 实例化即强制校验 embedding_func (不允许 None).
    注意:
    1. EmbeddingFunc 只按 func(texts) 调用, 无 kwargs 传递机制, 所有
       provider 配置 (model/base_url/api_key) 必须在此闭包内绑定.
    2. openai_embed 本身已是装饰后的 EmbeddingFunc 实例 (默认 dim=1536);
       必须用 openai_embed.func 取原始函数再包装, 避免嵌套 unwrap.
    3. free/paid 同模型同维度 (1024), 两种档生成的向量可混用.
    """
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        )(_embed_with_fallback),
    )